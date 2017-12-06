import os
import shutil
import argparse
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from process_data import save_pickle, load_pickle, load_task, load_processed_json, load_glove_weights
from process_data import to_var, make_vector
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
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')
args = parser.parse_args()

train_json, train_shared_json = load_processed_json('./dataset/data_train.json', './dataset/shared_train.json')
train_data = DataSet(train_json, train_shared_json)
ctx_maxlen = train_data.get_ctx_maxlen()
ctx_sent_maxlen, query_sent_maxlen = train_data.get_sent_maxlen()
# ctx_word_maxlen, query_word_maxlen = train_data.get_word_maxlen()
w2i, c2i = train_data.get_word_index()

print('----')
print('n_train', train_data.size())
# print('n_dev', len(dev_data))
print('ctx_maxlen', ctx_maxlen)
print('vocab_size_w:', len(w2i))
print('vocab_size_c:', len(c2i))
print('ctx_sent_maxlen:', ctx_sent_maxlen)
print('query_sent_maxlen:', query_sent_maxlen)
# print('ctx_word_maxlen:', ctx_word_maxlen)
# print('query_word_maxlen:', query_word_maxlen)

if args.use_pickle == 1:
    glove_embd_w = load_pickle('./pickle/glove_embd_w.pickle')
else:
    glove_embd_w = torch.from_numpy(load_glove_weights('./dataset', args.w_embd_size, len(w2i), w2i)).type(torch.FloatTensor)
    # save_pickle(glove_embd_w, './pickle/glove_embd_w.pickle')

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


def custom_loss_fn(data, labels):
    loss = Variable(torch.zeros(1))
    for d, label in zip(data, labels):
        loss -= torch.log(d[label]).cpu()
    loss /= data.size(0)
    return loss


def train(model, data, optimizer, n_epoch=10, batch_size=args.batch_size):
    print('----Train---')
    model.train()
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        batches = data.get_batches(batch_size, shuffle=True)
        p1_acc, p2_acc = 0, 0
        total = 0
        for i, batch in enumerate(tqdm(batches)):
            # (c, cc, q, cq, a)
            ctx_sent_len   = max([len(d[0]) for d in batch])
            ctx_word_len   = max([len(w) for d in batch for w in d[1]])
            query_sent_len = max([len(d[2]) for d in batch])
            query_word_len = max([len(w) for d in batch for w in d[3]])
            c, cc, q, cq, ans_var = make_vector(batch, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len)
            a_beg = ans_var[:, 0]
            a_end = ans_var[:, 1] - 1
            p1, p2 = model(c, cc, q, cq)
            # loss_p1 = nn.NLLLoss()(p1, a_beg)
            # loss_p2 = nn.NLLLoss()(p2, a_end)
            loss_p1 = custom_loss_fn(p1, a_beg)
            loss_p2 = custom_loss_fn(p2, a_end)
            p1_acc += torch.sum(a_beg == torch.max(p1, 1)[1]).data[0]
            p2_acc += torch.sum(a_end == torch.max(p2, 1)[1]).data[0]
            total += len(batch)
            if i % 50 == 0:
                rep_str = '[{}] Epoch {} {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}'
                print(rep_str.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                     epoch,
                                     100*i/len(batches),
                                     loss_p1.data[0],
                                     loss_p2.data[0]))
                acc_str = 'p1 acc: {:.3f}% ({}/{}), p2 acc: {:.3f}% ({}/{})'
                print(acc_str.format(100*p1_acc/total,
                                     p1_acc,
                                     total,
                                     100*p2_acc/total,
                                     p2_acc,
                                     total))

            model.zero_grad()
            (loss_p1+loss_p2).backward()
            # (loss_p1).backward()
            optimizer.step()

        # end eopch
        print('======== Epoch {} result ========'.format(epoch))
        print('p1 acc: {:.3f}, p2 acc: {:.3f}'.format(p1_acc/total, p2_acc/total))
        filename = '{}/Epoch-{}.model'.format('./checkpoints', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, True, filename=filename)


# test() {{{
def test(model, data, batch_size=args.batch_size):
    print('----Test---')
    model.eval()
    p1_acc, p2_acc = 0, 0
    total = 0
    batches = data.get_batches(batch_size)
    for i, batch in enumerate(tqdm(batches)):
        # (c, cc, q, cq, a)
        ctx_sent_len   = max([len(d[0]) for d in batch])
        ctx_word_len   = max([len(w) for d in batch for w in d[1]])
        query_sent_len = max([len(d[2]) for d in batch])
        query_word_len = max([len(w) for d in batch for w in d[3]])
        c, cc, q, cq, ans_var = make_vector(batch, w2i, c2i, ctx_sent_len, ctx_word_len, query_sent_len, query_word_len)
        a_beg = ans_var[:, 0]
        a_end = ans_var[:, 1] - 1
        p1, p2 = model(c, cc, q, cq)
        p1_acc += torch.sum(a_beg == torch.max(p1, 1)[1]).data[0]
        p2_acc += torch.sum(a_end == torch.max(p2, 1)[1]).data[0]
        total += batch_size

    print('======== Test result ========')
    print('p1 acc: {:.3f}%, p2 acc: {:.3f}%'.format(100*p1_acc/total, 100*p2_acc/total))
# }}}


model = AttentionNet(args)

if torch.cuda.is_available():
    print('use cuda')
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])

# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5) #, weight_decay=0.999)
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

if args.test == 1:
    test(model, train_data)
else:
    train(model, train_data, optimizer)
    # train(model, train_vec_data, optimizer)
print('finish')
