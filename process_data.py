import os
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize
import random
import torch
from torch.autograd import Variable


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def lower_list(str_list):
    return [str_var.lower() for str_var in str_list]


def load_task(dataset_path):
    ret_data = []
    ctx_max_len = 0 # character level length
    with open(dataset_path) as f:
        data = json.load(f)
        ver = data['version']
        print('dataset version:', ver)
        data = data['data']
        for i, d in enumerate(data):
            if i % 100 == 0:
                print('load_task:', i, '/', len(data))
            # print('load', d['title'], i, '/', len(data))
            for p in d['paragraphs']:
                if len(p['context']) > ctx_max_len:
                    ctx_max_len = len(p['context'])
                c = word_tokenize(p['context'])
                cc = [list(w) for w in c]
                q, a = [], []
                for qa in p['qas']:
                    q = word_tokenize(qa['question'])
                    qc = [list(w) for w in q]
                    a = [ans['text'] for ans in qa['answers']]
                    a_beg = [ans['answer_start'] for ans in qa['answers']]
                    a_end = [ans['answer_start'] + len(ans['text']) for ans in qa['answers']]
                    ret_data.append((c, cc, qa['id'], q, qc, a, a_beg, a_end)) # TODO context redandancy
    return ret_data, ctx_max_len


def load_processed_data(fpath):
    ctx_max_len = 0 # character level length
    with open(fpath) as f:
        lines = f.readlines()
        data = []
        for l in lines:
            c_label, c, q, a, a_txt = l.rstrip().split('\t')
            if len(c) > ctx_max_len:
                ctx_max_len = len(c)
            c, q, a = c.split(' '), q.split(' '), a.split(' ')
            # if len(c) > 30: continue # TMP
            c, q = lower_list(c), lower_list(q)
            cc = [list(w) for w in c]
            qc = [list(w) for w in q]
            a = [int(aa) for aa in a]
            a = [a[0], a[-1]]
            data.append((c_label, c, cc, q, qc, a, a_txt))
    return data, ctx_max_len


def load_processed_json(fpath_data, fpath_shared):
    # shared ------------
    # x: word level context list
    # cx: chara level context list
    # p: raw str level context list
    # word_counter: word to index
    # char_coun0ter: char to index
    # lower_word_counter: low word counter
    # word2vec: word2vec pretrained weights
    # lower_word2vec: lowered word2vec pretrained weights
    # data ------------
    # q: word level question
    # cq: char-word level question
    # y: word level id
    # *x: [article_id, paragraph_id]
    # *cx: same as *x
    # cy: ?
    # idxs: nothing meaning
    # ids: question id
    # answers: original answer text
    # *p: same as *x
    data = json.load(open(fpath_data))
    shared = json.load(open(fpath_shared))
    return data, shared


def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def make_word_vector(data, w2i_w, query_len):
    vec_data = []
    for sentence in data:
        index_vec = [w2i_w[w] for w in sentence]
        pad_len = max(0, query_len - len(index_vec))
        index_vec += [0] * pad_len
        index_vec = index_vec[:query_len]
        vec_data.append(index_vec)

    return to_var(torch.LongTensor(vec_data))


def make_char_vector(data, w2i_c, query_len, word_len):
    tmp = torch.zeros(len(data), query_len, word_len).type(torch.LongTensor)
    for i, words in enumerate(data):
        for j, word in enumerate(words):
            for k, ch in enumerate(word):
                tmp[i][j][k] = w2i_c[ch]
    return to_var(tmp)


class DataSet(object):
    def __init__(self, data, shared):
        self.data = data
        self.shared = shared
        self.NULL = "-NULL-"
        self.UNK = "-UNK-"
        self.ENT = "-ENT-"

    def size(self):
        return len(self.data['q'])

    def get_batches(self, batch_size, shuffle=False):
        # TODO shuffle
        batches = []
        for i in range(0, self.size()-batch_size, batch_size): # TODO shuffle, last elms
            batch = []
            for j in range(batch_size):
                q_idx = i + j
                rx = self.data['*x'][q_idx]
                c = self.shared['x'][rx[0]][rx[1]][0]
                q = self.data['q'][q_idx]
                a = self.data['y'][q_idx][0] # [[0, 80], [0, 82]] TODO only use 1-best
                a = (a[0][1], a[1][1]) # (80, 82) <= [[0, 80], [0, 82]]
                batch.append((c, q, a))
            batches.append(batch)
        if shuffle:
            random.shuffle(batches)
        return batches

    def get_ctx_maxlen(self):
        # char level context maxlen
        return max([len(p) for pp in self.shared['p'] for p in pp])

    def get_sent_maxlen(self):
        # word level sentence maxlen
        return max([len(articles[0]) for xx in self.shared['x'] for articles in xx]), max([len(q) for q in self.data['q']])

    def get_word_maxlen(self):
        # max word len
        return max([len(w) for xx in self.shared['x'] for articles in xx for w in articles[0]]), max([len(w) for q in self.data['q'] for w in q])

    def get_word_index(self, word_count_th=10, char_count_th=100):

        word2vec_dict = self.get_word2vec()
        word_counter = self.get_word_counter()
        char_counter = self.get_char_counter()
        w2i = {w: i+3 for i, w in enumerate(w for w, ct in word_counter.items()
                                            if ct > word_count_th or (w in word2vec_dict))}
        c2i = {c: i+2 for i, c in
                    enumerate(c for c, ct in char_counter.items()
                              if ct > char_count_th)}
        w2i[self.NULL] = 0
        w2i[self.UNK] = 1
        w2i[self.ENT] = 2
        c2i[self.NULL] = 0
        c2i[self.UNK] = 1
        c2i[self.ENT] = 2

        return w2i, c2i

    def get_word2vec(self):
        return self.shared['lower_word2vec']

    def get_word_counter(self):
        return self.shared['lower_word_counter']

    def get_char_counter(self):
        return self.shared['char_counter']

    def _make_word_vector(self, sentence, w2i, seq_len):
        index_vec = [w2i[w] if w in w2i else w2i[self.UNK] for w in sentence]
        pad_len = max(0, seq_len - len(index_vec))
        index_vec += [w2i[self.NULL]] * pad_len
        index_vec = index_vec[:seq_len]
        return index_vec

    def make_word_vector(self, batch_data, w2i, ctx_len, query_len):
        context, query, ans = [], [], []
        for c, q, a in batch_data:
            context.append(self._make_word_vector(c, w2i, ctx_len))
            query.append(self._make_word_vector(q, w2i, query_len))
            ans.append(a)
        return to_var(torch.LongTensor(context)), to_var(torch.LongTensor(query)), to_var(torch.LongTensor(ans))
