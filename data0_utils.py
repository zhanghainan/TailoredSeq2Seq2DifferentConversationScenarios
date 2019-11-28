import sys

import random
import numpy as np

Letor07Path = '/home/zhanghainan/tensorflow/DeepRL4Dialogue-weibo-rl-with-baseline/OpenSubData/discrimiterData'
sys.path.insert(0, '/home/zhanghainan/')

import pytextnet as pt

word_dict, iword_dict = pt.io.base.read_word_dict(filename=Letor07Path + '/word.txt')
query_data = pt.io.base.read_data(filename=Letor07Path + '/qid_query.txt')
doc_data = pt.io.base.read_data(filename=Letor07Path + '/replyid_reply.txt')
embed_dict = pt.io.base.read_embedding(filename=Letor07Path + '/embedding.txt')

_PAD_ = len(word_dict)
embed_dict[_PAD_] = np.zeros((50, ), dtype=np.float32)
word_dict[_PAD_] = '[PAD]'
iword_dict['[PAD]'] = _PAD_
W_init_embed = np.float32(np.random.uniform(-0.02, 0.02, [len(word_dict), 50]))
embedding = pt.io.base.convert_embed_2_numpy(embed_dict, embed = W_init_embed)

class PairGenerator():
    def __init__(self, rel_file, config):
        rel = pt.io.base.read_relation(filename=rel_file)
        self.pair_list = self.make_pair(rel)
        self.config = config

    def make_pair(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse = True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx+1:]:
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append( (d1, high_d2, low_d2) )
        print 'Pair Instance Count:', len(pair_list)
        return pair_list

    def get_batch(self, data1, data2):
        config = self.config
        X1 = np.zeros((config['batch_size']*2, config['data1_maxlen']), dtype=np.int32)
        X1_len = np.zeros((config['batch_size']*2,), dtype=np.int32)
        X2 = np.zeros((config['batch_size']*2, config['data2_maxlen']), dtype=np.int32)
        X2_len = np.zeros((config['batch_size']*2,), dtype=np.int32)
        Y = np.zeros((config['batch_size']*2,), dtype=np.int32)

        Y[::2] = 1
        X1[:] = config['fill_word']
        X2[:] = config['fill_word']
        for i in range(config['batch_size']):
            d1, d2p, d2n = random.choice(self.pair_list)
            d1_len = min(config['data1_maxlen'], len(data1[d1]))
            d2p_len = min(config['data2_maxlen'], len(data2[d2p]))
            d2n_len = min(config['data2_maxlen'], len(data2[d2n]))
            X1[i*2,   :d1_len],  X1_len[i*2]   = data1[d1][:d1_len],   d1_len
            X2[i*2,   :d2p_len], X2_len[i*2]   = data2[d2p][:d2p_len], d2p_len
            X1[i*2+1, :d1_len],  X1_len[i*2+1] = data1[d1][:d1_len],   d1_len
            X2[i*2+1, :d2n_len], X2_len[i*2+1] = data2[d2n][:d2n_len], d2n_len

        return X1, X1_len, X2, X2_len, Y

class ListGenerator():
    def __init__(self, rel_file, config):
        rel = pt.io.base.read_relation(filename=rel_file)
        self.list_list = self.make_list(rel)
        self.config = config

    def make_list(self, rel):
        list_list = {}
        for label, d1, d2 in rel:
            if d1 not in list_list:
                list_list[d1] = []
            list_list[d1].append( (label, d2) )
        for d1 in list_list:
            list_list[d1] = sorted(list_list[d1], reverse = True)
        print 'List Instance Count:', len(list_list)
        return list_list.items()

    def get_batch(self, data1, data2):
        config = self.config
        for i, (d1, d2_list) in enumerate(self.list_list):
            X1 = np.zeros((len(d2_list), config['data1_maxlen']), dtype=np.int32)
            X1_len = np.zeros((len(d2_list),), dtype=np.int32)
            X2 = np.zeros((len(d2_list), config['data2_maxlen']), dtype=np.int32)
            X2_len = np.zeros((len(d2_list),), dtype=np.int32)
            Y = np.zeros((len(d2_list),), dtype= np.int32)
            X1[:] = config['fill_word']
            X2[:] = config['fill_word']
            d1_len = min(config['data1_maxlen'], len(data1[d1]))
            for j, (l, d2) in enumerate(d2_list):
                d2_len = min(config['data2_maxlen'], len(data2[d2]))
                X1[j, :d1_len], X1_len[j] = data1[d1][:d1_len], d1_len
                X2[j, :d2_len], X2_len[j] = data2[d2][:d2_len], d2_len
                Y[j] = l
            yield X1, X1_len, X2, X2_len, Y
