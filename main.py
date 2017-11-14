import numpy as np
import os
import json
from nltk.tokenize import word_tokenize
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchtext

use_cuda = torch.cuda.is_available()
from process_data import save_pickle, load_pickle, load_task, load_glove_weights
from process_data import to_var, make_word_vector, make_char_vector
from layers.char_embedding import CharEmbedding
from layers.word_embedding import WordEmbedding
from layers.highway import Highway
from layers.attention_net import AttentionNet
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--embd_size', type=int, default=100, help='embedding size')
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
print(args)

train_data, train_ctx_maxlen = load_task('./dataset/train-v1.1.json')
dev_data, dev_ctx_maxlen = load_task('./dataset/dev-v1.1.json')
data = train_data + dev_data
ctx_maxlen = max(train_ctx_maxlen, dev_ctx_maxlen)
save_pickle(train_data, 'pickle/train_data.pickle')
save_pickle(dev_data, 'pickle/dev_data.pickle')

vocab_w, vocab_c = set(), set()
for ctx_w, ctx_c, q_id, q_w, q_c, answer, _, _ in data:
    vocab_w |= set(ctx_w + q_w + answer)
    flatten_c = [c for chars in ctx_c for c in chars]
    flatten_q = [c for chars in q_c for c in chars]

    vocab_c |= set(flatten_c + flatten_q) # TODO

vocab_w = list(sorted(vocab_w))
vocab_c = list(sorted(vocab_c))

w2i_w = dict((w, i) for i, w in enumerate(vocab_w, 0))
i2w_w = dict((i, w) for i, w in enumerate(vocab_w, 0))
w2i_c = dict((c, i) for i, c in enumerate(vocab_c, 0))
i2w_c = dict((i, c) for i, c in enumerate(vocab_c, 0))
save_pickle(vocab_w, 'pickle/vocab_w.pickle')
save_pickle(vocab_c, 'pickle/vocab_c.pickle')
save_pickle(w2i_w, 'pickle/w2i_w.pickle')
save_pickle(w2i_c, 'pickle/w2i_c.pickle')
save_pickle(i2w_w, 'pickle/i2w_w.pickle')
save_pickle(i2w_c, 'pickle/i2w_c.pickle')
# train_data = load_pickle('pickle/train_data.pickle')
# vocab = load_pickle('pickle/vocab.pickle')
# w2i = load_pickle('pickle/w2i.pickle')

vocab_size_w = len(vocab_w)
vocab_size_c = len(vocab_c)

ctx_sent_maxlen = max([len(c) for c, _, _, _, _, _, _, _ in data])
query_sent_maxlen = max([len(q) for _, _, _, q, _, _, _, _ in data])
ctx_word_maxlen = max([len(w) for _, cc, _, _, _, _, _, _ in data for w in cc])
query_word_maxlen = max([len(w) for _, _, _, _, qc, _, _, _ in data for w in qc])
print('----')
print('n_train', len(train_data))
# print('n_dev', len(dev_data))
print('ctx_maxlen', ctx_maxlen)
print('vocab_size_w:', vocab_size_w)
print('vocab_size_c:', vocab_size_c)
print('ctx_sent_maxlen:', ctx_sent_maxlen)
print('query_sent_maxlen:', query_sent_maxlen)
print('ctx_word_maxlen:', ctx_word_maxlen)
print('query_word_maxlen:', query_word_maxlen)


glove_embd_w = torch.from_numpy(load_glove_weights('./dataset', args.embd_size, vocab_size_w, w2i_w))

# args = {
#     'embd_size': embd_size,
#     'vocab_size_c': vocab_size_c,
#     'vocab_size_w': vocab_size_w,
#     'pre_embd_w': glove_embd_w, # word embedding
#     'filters': [[1, 5]], # char embedding
#     'out_chs': 100, # char embedding
#     'ans_size': ctx_maxlen
# }
# args = Config(**args)
args.vocab_size_c = vocab_size_c
args.vocab_size_w = vocab_size_w
args.pre_embd_w = glove_embd_w
args.filters = [[1, 5]]
args.out_chs = 100
args.ans_size = ctx_maxlen
        
def train(model, optimizer, n_epoch=10, batch_size=8):
    for epoch in range(n_epoch):
        for i in range(0, len(data)-batch_size, batch_size): # TODO shuffle, last elms
            print('batch', i, '/', len(data))
            batch_data = data[i:i+batch_size]
            c = [d[0] for d in batch_data]
            cc = [d[1] for d in batch_data]
            q = [d[3] for d in batch_data]
            qc = [d[4] for d in batch_data]
            a_beg = to_var(torch.LongTensor([d[6] for d in batch_data]).squeeze())
            a_end = to_var(torch.LongTensor([d[7] for d in batch_data]).squeeze())
            c_char_var = make_char_vector(cc, w2i_c, ctx_sent_maxlen, ctx_word_maxlen)
            c_word_var = make_word_vector(c, w2i_w, ctx_sent_maxlen)
            q_char_var = make_char_vector(qc, w2i_c, query_sent_maxlen, query_word_maxlen)
            q_word_var = make_word_vector(q, w2i_w, query_sent_maxlen)
            p1, p2 = model(c_char_var, c_word_var, q_char_var, q_word_var)
            loss_p1 = nn.NLLLoss()(p1, a_beg)
            loss_p2 = nn.NLLLoss()(p2, a_end)
            print(loss_p1.data, loss_p2.data)
            model.zero_grad()
            (loss_p1+loss_p2).backward()
            optimizer.step()
            
model = AttentionNet(args)
if torch.cuda.is_available():
    model.cuda()
# print(model)
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5)
train(model, optimizer)
print('finish train')


