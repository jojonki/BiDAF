import os
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize
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
