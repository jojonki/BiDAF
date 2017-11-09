import os
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize

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
    with open(dataset_path) as f:
        data = json.load(f)
        ver = data['version']
        print('dataset version:', ver)
        data = data['data']
        for i, d in enumerate(data):
            if i % 100 == 0: print('load_task:', i, '/', len(data))
            # print('load', d['title'], i, '/', len(data))
            for p in d['paragraphs']:
                c = word_tokenize(p['context'])
                cc = [list(w) for w in c]
                q, a = [], []
                for qa in p['qas']:
                    q = word_tokenize(qa['question'])
                    qc = [list(w) for w in q]
                    a = [ans['text'] for ans in qa['answers']]
                    ret_data.append((c, cc, qa['id'], q, qc, a)) # TODO context redandancy
#                 break
            break
    return ret_data


def vectorize(data, w2i, ctx_maxlen, qst_maxlen):
    C, Q, A = [], [], []
    for i, (context, _, question, answer) in enumerate(data):
        if i % 10000 == 0: print('vectroize:', i, '/', len(data))
        # not use context
#         c = [w2i[w] for w in context if w in w2i]
#         c = c[:ctx_maxlen]
#         c_pad_len = max(0, ctx_maxlen - len(c))
#         c += [0] * c_pad_len

        q = [w2i[w] for w in question if w in w2i]
        q = q[:qst_maxlen]
        q_pad_len = max(0, qst_maxlen - len(q))
        q += [0] * q_pad_len

        y = np.zeros(len(w2i))
        if answer[0] in w2i:
            y[w2i[answer[0]]] = 1

#         C.append(c)
        Q.append(q)
        A.append(y)
    
#     C = np.array(C)#, dtype=np.uint32)
    Q = np.array(Q)
    A = np.array(A, dtype='byte')

    return C, Q, A

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
