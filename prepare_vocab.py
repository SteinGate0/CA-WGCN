"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter

from utils import vocab, constant, helper

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='Dataset/MyDataset',help='MyDataset directory.')
    parser.add_argument('--vocab_dir', default='Dataset/vocab',help='Output vocab directory.') #embedding.npy vocal.pkl的输出路径
    parser.add_argument('--Baidu_dir', default='Dataset/baidu', help='Chinese Embedding directory.')
    parser.add_argument('--wv_file', default='sgns.baidubaike.txt', help='Chinese Embedding vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='baidubaike vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    #input file
    train_file = args.data_dir + '/train.json'
    dev_file = args.data_dir + '/dev.json'
    test_file = args.data_dir + '/test.json'
    wv_file = args.Baidu_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)   #Output vocab directory 不存在，创建一个文件夹
    vocab_file = args.vocab_dir + '/vocab.pkl'
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file)
    test_tokens = load_tokens(test_file)

    # load glove
    print("loading baidu...")
    baidu_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from baidu.".format(len(baidu_vocab)))

    print("building vocab...")
    v = build_vocab(train_tokens, baidu_vocab, args.min_freq)

    print("calculating oov...")  #oov: Out-of-vocabulary
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))

    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")
    print(v)

def load_tokens(filename):
    with open(filename,encoding='utf-8') as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['tokens']
            ss, se, os, oe = d['e1_pos'], d['e1_pos'], d['e2_pos'], d['e2_pos']
            # do not create vocab for entity words
            ts[ss:se+1] = ['<PAD>']*(se-ss+1)
            ts[os:oe+1] = ['<PAD>']*(oe-os+1)
            tokens += list(filter(lambda t: t!='<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, baidu_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in baidu_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + entity_masks() + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def entity_masks():
    """ Get all entity mask tokens as a list. """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

if __name__ == '__main__':
    main()