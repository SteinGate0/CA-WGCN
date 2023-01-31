"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):   #inputs is list ,length is 10 ,[words(20,61), masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type(20,1)]
        outputs, pooling_output = self.gcn_model(inputs)   #outputs.shape=pooling_output.shape = [b,200]
        logits = self.classifier(outputs)   #logits.shape = [b,10]
        return logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])  #hidden_dim = 200,num_layers=2

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):         #mlp_layers = 2
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs): #inputs is list ,length is 10 ,[words(20,61), masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type(20,1)]
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)   #将mask矩阵中的True/Fasle->1/0,记录每个batch有多少个单词
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)  # axis=0 跨行进行操作  shape = [b，maxlen,maxlen]
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        #.data用法可以修改tensor的值而不被autograd(不会影响反向传播)，
        # subj_pos,obj_pos均为主语谓语在句子中的位置，#返回距离List :[-3,-2,-1,0,0,0,1,2,3]
        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs) #将此batch的adj邻接矩阵，与输入输入到gcn,
        # 得到#h = gcn_inputs:[b,maxlen,200]  ,pool_mask=mask:[b,maxlen,1]

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        #第一个eq(0)实体位置都标记为True其余位置标记为False,第二个eq(0),实体位置标记为False,其余位置标记为False  shape:[b,maxlen,1]
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)# shape:[b,200]
        subj_out = pool(h, subj_mask, type=pool_type) #shape:[b,200]
        obj_out = pool(h, obj_mask, type=pool_type) #shape:[b,200]
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)  #[b,600]
        outputs = self.out_mlp(outputs)   #shape:[b,200]
        return outputs, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers    #num_layers = 2
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim  #hidden_dim = 200
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']  #300+30+30

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):    #如果是C-GCN
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)   #rnn_layers = 1
            self.in_dim = opt['rnn_hidden'] * 2  #400
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            #有Rnn的情况下如果是第一层，则输入为BiLSTM的400维hidden
            #无Rnn的情况下第一层，300+30+30
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):   #卷积层参数通过权重衰减防止过拟合
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size): #rnn_input=embs=[40,max(len(x)),360]
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())  #bacth中每句话的长度
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])#shape:[2,40,200]  rnn_layers = 1 run_hidden=200
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True) #
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)#[40,max(len(x)),400]
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack  mask.shape:[batch,maxlen]
        word_embs = self.emb(words)  #shape:[40,max(len(x)),300]
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2) #shape:[40,max(len(x)),300+30+30]
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0])) #从RNN层出来了shape=[40,max(len(x)),400]
        else:
            gcn_inputs = embs  ##shape:[40,max(len(x)),300+30+30]
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        #[b,maxlen,maxlen]->[b,maxlen]->[b,maxlen,1]再所有位置数值＋1反正0数据无法除,我理解的含义是，adj.sum(2)每个节点都多少条出边记录
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)   #adj.sum(1)每个节点有多少个入边
        #[b,maxlen]+[b,maxlen] = [b,maxlen]等于0的位置全为True,不等于0的位置为False,[b,maxlen,1]
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        #下面紧扣GCN公式还原
        for l in range(self.layers):  #self.layers = 2
            Ax = adj.bmm(gcn_inputs)  #[b，maxlen,maxlen]*[40,max(len(x)),400]或者[b,max(len(x)),300+30+30] = [b,maxlen,200/360]
            AxW = self.W[l](Ax)     #第一层：W[1]:nn.Linear(400/360, 200)  AxW.shape=[b,maxlen,200]
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            # 这代码写得太好了，前面AxW是WAh ，后边是Wh ,两者相加可以把Wh提出来:Wh（A+I）符合了论文中提到的公式
            AxW = AxW / denom # 节点i在图中的度

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW #除最后一层外都dropout

        return gcn_inputs, mask   #gcn_inputs:[b,maxlen,200]  ,mask:[b,maxlen,1]

def pool(h, mask, type='max'):    #h:[b,maxlen,200]    mask:[b,maxlen,1]
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)   #使用value填充mask中为1位置的元素
        return torch.max(h, 1)[0]  #只返回最大值的每个数 ,1且是按照行为单位即从[maxlen,200中]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)   #shape:[2,40,200]
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

