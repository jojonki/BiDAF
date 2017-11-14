import numpy as np
import os
import shutil
import json
from nltk.tokenize import word_tokenize
import argparse
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchtext

from process_data import save_pickle, load_pickle, load_task, load_glove_weights
from process_data import to_var, make_word_vector, make_char_vector
from layers.char_embedding import CharEmbedding
from layers.word_embedding import WordEmbedding
from layers.highway import Highway
from layers.attention_net import AttentionNet
# from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--w_embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--c_embd_size', type=int, default=8, help='character embedding size')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--use_pickle', type=int, default=1, help='load dataset from pickles')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')

args = parser.parse_args()

if args.use_pickle == 1:
    train_data = load_pickle('pickle/train_data.pickle')
    dev_data = load_pickle('pickle/dev_data.pickle')
    data = train_data + dev_data
    ctx_maxlen = 4063 #TODO

    vocab_w = load_pickle('pickle/vocab_w.pickle')
    vocab_c = load_pickle('pickle/vocab_c.pickle')
    w2i_w = load_pickle('pickle/w2i_w.pickle')
    i2w_w = load_pickle('pickle/i2w_w.pickle')
    w2i_c = load_pickle('pickle/w2i_c.pickle')
    i2w_c = load_pickle('pickle/i2w_c.pickle')
else:
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

if args.use_pickle == 1:
    glove_embd_w = load_pickle('./pickle/glove_embd_w.pickle')
else:
    glove_embd_w = torch.from_numpy(load_glove_weights('./dataset', args.w_embd_size, vocab_size_w, w2i_w)).type(torch.FloatTensor)
    save_pickle(glove_embd_w, './pickle/glove_embd_w.pickle')
    
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
print(args)

def save_checkpoint(state, is_best, filename='./checkpoints/checkpoint.pth.tar'):
    print('save model!!!!')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.resume)
        
def batch_ranking(p1, p2):
    batch_size = p1.size(0)
    p1_rank, p2_rank = [], []
    for i in range(batch_size):
        p1_rank.append(sorted(range(len(p1[i])), key=lambda k: p1[i][k].data[0], reverse=True))
        p2_rank.append(sorted(range(len(p2[i])), key=lambda k: p2[i][k].data[0], reverse=True))
    return p1_rank, p2_rank
        
def train(model, optimizer, n_epoch=10, batch_size=args.batch_size):
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        for i in range(0, len(data)-batch_size, batch_size): # TODO shuffle, last elms
            batch_data = data[i:i+batch_size]
            c = [d[0] for d in batch_data]
            cc = [d[1] for d in batch_data]
            q = [d[3] for d in batch_data]
            qc = [d[4] for d in batch_data]
            a_beg = to_var(torch.LongTensor([d[6][0] for d in batch_data]).squeeze()) # TODO: multi target
            a_end = to_var(torch.LongTensor([d[7][0] for d in batch_data]).squeeze()) 
            c_char_var = make_char_vector(cc, w2i_c, ctx_sent_maxlen, ctx_word_maxlen)
            c_word_var = make_word_vector(c, w2i_w, ctx_sent_maxlen)
            q_char_var = make_char_vector(qc, w2i_c, query_sent_maxlen, query_word_maxlen)
            q_word_var = make_word_vector(q, w2i_w, query_sent_maxlen)
            p1, p2 = model(c_char_var, c_word_var, q_char_var, q_word_var)
            loss_p1 = nn.NLLLoss()(p1, a_beg)
            loss_p2 = nn.NLLLoss()(p2, a_end)
            if i % 100 == 0:
                now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                print('[{}] {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}'.format(now, 100*i/len(data), loss_p1.data[0], loss_p2.data[0]))
                p1_rank, p2_rank = batch_ranking(p1, p2)
                for rank in range(1): # N-best, currently 1-best
                    p1_rank_id = p1_rank[0][rank]
                    p2_rank_id = p2_rank[0][rank]
                    print('Rank {}, p1_result={}, p2_result={}'.format(
                        rank+1, p1_rank_id==a_beg.data[0], p2_rank_id==a_end.data[0]))
                # TODO calc acc, save every epoch wrt acc

            model.zero_grad()
            (loss_p1+loss_p2).backward()
            optimizer.step()
        
        # end eopch
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, True)

def test(model, batch_size=args.batch_size):
    pass

model = AttentionNet(args)
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5, weight_decay=0.999)
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume)) 

if torch.cuda.is_available():
    model.cuda()

print(model)
train(model, optimizer)
print('finish train')


