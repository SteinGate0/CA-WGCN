"""
Basic operations on trees.
"""
import math
import numpy as np
from collections import defaultdict

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    stanfordnlp/treelstm重用的树对象
    """
    def __init__(self):
        self.dist = 0
        self.idx = 0
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    将head索引序列转换为tree对象
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None

    # find dependency path
    subj_pos = [i for i in range(len_) if subj_pos[i] == 0]  #subj_pos为0的部分实际上是实体，返回主语实体的下标[3,4,5]
    obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

    cas = None

    subj_ancestors = set(subj_pos)
    for s in subj_pos:      #遍历主语实体的每一个下标
        h = head[s]
        tmp = [s]
        while h > 0:        #head如果不是root
            tmp += [h-1]    #tmp存储当前节点s与发射边节点（head,s），以及发射边节点的祖先
            subj_ancestors.add(h-1)
            #subj_ancestors存储主语实体下标除root之外的所有发射边节点，以及发射边节点的祖先，一直到找到root根节停止
            h = head[h-1]

        if cas is None:
            cas = set(tmp)      #第一次遍历cas是空的，把第一个下标节点对应的所有祖先加入
        else:
            cas.intersection_update(tmp)  #第二三次遍历就调用intersection_update取交集，最后保留几个主语实体节点的公共祖先

    obj_ancestors = set(obj_pos)
    for o in obj_pos:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h-1]
            obj_ancestors.add(h-1)
            h = head[h-1]
        cas.intersection_update(tmp)    #cas再与宾语实体节点的公共祖先取交集

    # find lowest common ancestor
    if len(cas) == 1:           #只有一个公共节点那么LCA就是它
        lca = list(cas)[0]
    else:
        child_count = {k:0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:  #ca的祖先不是根节点 and ca的祖先在cas这堆祖先节点中
                child_count[head[ca] - 1] += 1      #ca的祖先加一个孩子，那个孩子就是ca

        #LCA(Least Common Ancestors)
        # the LCA has no child in the CA set
        for ca in cas:          #很容易理解，公共祖先树中没孩子的‘叶子’肯定是所有实体节点的lCA
            if child_count[ca] == 0:
                lca = ca
                break

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    #主语树（含祖先）与宾语树（含祖先）取并集，再去掉公共祖先节点
    path_nodes.add(lca)   #再加上最低公共祖先，LCA树构造完成了

    # compute distance to path_nodes
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)]#LCA树中的节点被标记为0，其他节点标记为-1

    for i in range(len_):
        if dist[i] < 0:         #如果不是LCA的节点
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:#
                stack.append(head[stack[-1]] - 1)   #stack存储节点i以及他的祖先们，直到最高的祖先在path_nodes中

            if stack[-1] in path_nodes:             #如果节点i的最高祖先在LCA中
                for d, j in enumerate(reversed(stack)):  #stack存储的路径反序i<-B<-A 变成 A->B->i
                    dist[j] = d             #dist[A] = 0 ,dist[B] = 1,dist[i] = 2，显然dist表示了各个节点到LCA树的距离
            else:
                for j in stack:             #这部分节点说明与LCA没有边连接到，与LCA的距离自然是无穷大
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4) # aka infinity

    highest_node = lca
    nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]  #剪枝 prune<=k,满足要求的节点Tree()，不满足要求的为None

    #遍历一遍nodes，将LCA树创建好
    for i in range(len(nodes)):
        if nodes[i] is None:
            continue
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = dist[i]
        if h > 0 and i != highest_node:
            assert nodes[h-1] is not None
            nodes[h-1].add_child(nodes[i])

    root = nodes[highest_node]

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    把一个树对象转为邻接矩阵
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    k = 1
    queue = [tree]   #树LCA根节点入队
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]     #LCA树的节点编号

        for c in t.children:    #t节点有孩子c节点，所以t到c有临界边
            ret = get_adjElement(t, c, ret, k)
            #ret[t.idx, c.idx] = 1
        queue += t.children     #孩子节点入队遍历

    if not directed:            #这个参数关键决定是双向还是单向图
        ret = ret + ret.T

    if self_loop:               #节点到自身循环边
        for i in idx:
            ret[i, i] = 1

    return ret


#递归find节点高度
def get_adjElement(rootNode, otherNode , ret, k):#遍历当前节点下的所有子节点(otherNode表示)，k表示rootNode和otherNode之间的距离
    adj = ret
    adj[rootNode.idx, otherNode.idx] = 1 / math.e**(k-1)
    if otherNode.num_children != 0:#otherNode是叶子节点
        k = k+1
        for c in otherNode.children:
            adj = get_adjElement(rootNode, c, adj, k)

    return adj

def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret