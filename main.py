import os
import shutil
import argparse
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

from process_data import save_pickle, load_pickle, load_task, load_processed_json, load_glove_weights
from process_data import to_var, make_word_vector, make_char_vector
from process_data import DataSet
from layers.attention_net import AttentionNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--w_embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--c_embd_size', type=int, default=8, help='character embedding size')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--use_pickle', type=int, default=0, help='load dataset from pickles')
parser.add_argument('--test_mode', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')
args = parser.parse_args()

train_data, train_shared = load_processed_json('./dataset/data_train.json', './dataset/shared_train.json')
train_ds = DataSet(train_data, train_shared)
ctx_maxlen = train_ds.get_ctx_maxlen()
ctx_sent_maxlen, query_sent_maxlen = train_ds.get_sent_maxlen()
ctx_word_maxlen, query_word_maxlen = train_ds.get_word_maxlen()
w2i, c2i = train_ds.get_word_index()

print('----')
print('n_train', len(train_data))
# print('n_dev', len(dev_data))
print('ctx_maxlen', ctx_maxlen)
print('vocab_size_w:', len(w2i))
print('vocab_size_c:', len(c2i))
print('ctx_sent_maxlen:', ctx_sent_maxlen)
print('query_sent_maxlen:', query_sent_maxlen)
print('ctx_word_maxlen:', ctx_word_maxlen)
print('query_word_maxlen:', query_word_maxlen)

if args.use_pickle == 1:
    glove_embd_w = load_pickle('./pickle/glove_embd_w.pickle')
else:
    glove_embd_w = torch.from_numpy(load_glove_weights('./dataset', args.w_embd_size, len(w2i), w2i)).type(torch.FloatTensor)
    save_pickle(glove_embd_w, './pickle/glove_embd_w.pickle')

args.vocab_size_c = len(c2i)
args.vocab_size_w = len(w2i)
# args.pre_embd_w = dt.get_word2vec()
args.pre_embd_w = glove_embd_w
args.filters = [[1, 5]]
args.out_chs = 100
args.ans_size = ctx_sent_maxlen
# print('---arguments---')
# print(args)


def save_checkpoint(state, is_best, filename='./checkpoints/checkpoint.pth.tar'):
    print('save model!', filename)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, args.resume)


# def batch_ranking(p1, p2):
#     batch_size = p1.size(0)
#     p1_rank, p2_rank = [], []
#     for i in range(batch_size):
#         p1_rank.append(sorted(range(len(p1[i])), key=lambda k: p1[i][k].data[0], reverse=True))
#         p2_rank.append(sorted(range(len(p2[i])), key=lambda k: p2[i][k].data[0], reverse=True))
#     return p1_rank, p2_rank


def train(model, data, optimizer, n_epoch=10, batch_size=args.batch_size):
    model.train()
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        # for i in range(0, len(train_data)-batch_size, batch_size): # TODO shuffle, last elms
        batches = data.get_batches(batch_size)
        for i, batch in enumerate(tqdm(batches)):
            c_word_var, q_word_var, ans_var = data.make_word_vector(batch, w2i, ctx_sent_maxlen, query_sent_maxlen)
            a_beg = ans_var[:, 0]
            # a_end = ans_var[:, 1]
            p1, p2 = model(None, c_word_var, None, q_word_var)
            loss_p1 = nn.NLLLoss()(p1, a_beg)
            # loss_p2 = nn.NLLLoss()(p2, a_end)
            if i % (batch_size*20) == 0:
                now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                print('[{}] Epoch {} {:.1f}%, loss_p1: {:.3f}'.format(now, epoch, 100*i/data.size(), loss_p1.data[0]))
                print('Acc:', torch.sum(a_beg == torch.max(p1, 1)[1]).data[0], '/', batch_size)
                # TODO calc acc, save every epoch wrt acc

            model.zero_grad()
            # (loss_p1+loss_p2).backward()
            loss_p1.backward()
            optimizer.step()

        # end eopch
        filename = '{}/Epoch-{}.model'.format('./checkpoints', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, True, filename=filename)


# test()
def test(model, data, batch_size=args.batch_size):
    model.eval()
    p1_acc_count = 0
    p2_acc_count = 0
    total = 0
    for i in tqdm(range(0, data.size()-batch_size, batch_size)): # TODO last elms
        batches = data.get_batches(batch_size)
        for i, batch in enumerate(tqdm(batches)):
            c_word_var, q_word_var, ans_var = data.make_word_vector(batch, w2i, ctx_sent_maxlen, query_sent_maxlen)
            a_beg = ans_var[:, 0]
            # a_end = ans_var[:, 1]
            p1, p2 = model(None, c_word_var, None, q_word_var)
            p1_acc_count += torch.sum(a_beg == torch.max(p1, 1)[1]).data[0]
            total += batch_size

    print('======== Test result ========')
    print('p1 acc: {:.3f}, p2 acc: {:.3f}'.format(p1_acc_count/total, p2_acc_count/total))


model = AttentionNet(args)

if torch.cuda.is_available():
    print('use cuda')
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])

# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5, weight_decay=0.999)
# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer']) # TODO ?
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
print('parameters-----')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())

if args.test_mode == 1:
    test(model, train_ds)
else:
    train(model, train_ds, optimizer)
print('finish')
