import os
import shutil
import argparse
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

from process_data import save_pickle, load_pickle, load_task, load_processed_data, load_glove_weights
from process_data import to_var, make_word_vector, make_char_vector
from layers.attention_net import AttentionNet

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--w_embd_size', type=int, default=100, help='word embedding size')
parser.add_argument('--c_embd_size', type=int, default=8, help='character embedding size')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--use_pickle', type=int, default=0, help='load dataset from pickles')
parser.add_argument('--test_mode', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--resume', default='./checkpoints/model_best.tar', type=str, metavar='PATH', help='path saved params')
args = parser.parse_args()


train_data, train_ctx_maxlen = load_processed_data('./dataset/train.txt')
dev_data, dev_ctx_maxlen = load_processed_data('./dataset/dev.txt')
data = train_data + dev_data
ctx_maxlen = max(train_ctx_maxlen, dev_ctx_maxlen)

vocab_w, vocab_c = set(), set()
for _, c, cc, q, qc, _, _ in data: # (c_label, c, cc, q, qc, a, a_txt)
    vocab_w |= set(c + q)
    flatten_c = [c for chars in cc for c in chars]
    flatten_q = [c for chars in qc for c in chars]
    vocab_c |= set(flatten_c + flatten_q) # TODO
vocab_w = list(sorted(vocab_w))
vocab_c = list(sorted(vocab_c))
vocab_size_w = len(vocab_w)
vocab_size_c = len(vocab_c)

w2i_w = dict((w, i) for i, w in enumerate(vocab_w, 0))
i2w_w = dict((i, w) for i, w in enumerate(vocab_w, 0))
w2i_c = dict((c, i) for i, c in enumerate(vocab_c, 0))
i2w_c = dict((i, c) for i, c in enumerate(vocab_c, 0))
ctx_sent_maxlen   = max([len(d[1]) for d in data])
query_sent_maxlen = max([len(d[3]) for d in data])
ctx_word_maxlen   = max([len(w) for d in data for w in d[1]])
query_word_maxlen = max([len(w) for d in data for w in d[3]])

print('----')
print('n_train', len(train_data))
print('n_dev', len(dev_data))
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

args.vocab_size_c = vocab_size_c
args.vocab_size_w = vocab_size_w
args.pre_embd_w = glove_embd_w
args.filters = [[1, 5]]
args.out_chs = 100
args.ans_size = ctx_sent_maxlen
print('---arguments---')
print(args)


def save_checkpoint(state, is_best, filename='./checkpoints/checkpoint.pth.tar'):
    print('save model!', filename)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, args.resume)


def batch_ranking(p1, p2):
    batch_size = p1.size(0)
    p1_rank, p2_rank = [], []
    for i in range(batch_size):
        p1_rank.append(sorted(range(len(p1[i])), key=lambda k: p1[i][k].data[0], reverse=True))
        p2_rank.append(sorted(range(len(p2[i])), key=lambda k: p2[i][k].data[0], reverse=True))
    return p1_rank, p2_rank


def train(model, optimizer, n_epoch=10, batch_size=args.batch_size):
    model.train()
    for epoch in range(n_epoch):
        print('---Epoch', epoch)
        for i in range(0, len(train_data)-batch_size, batch_size): # TODO shuffle, last elms
            # print('----------batch', i)
            batch_data = train_data[i:i+batch_size]
            # (c_label, c, cc, q, qc, a, a_txt)
            c = [d[1] for d in batch_data]
            cc = [d[2] for d in batch_data]
            q = [d[3] for d in batch_data]
            qc = [d[4] for d in batch_data]
            a_beg = to_var(torch.LongTensor([d[5][0] for d in batch_data]).squeeze()) # TODO: multi target
            a_end = to_var(torch.LongTensor([d[5][1] for d in batch_data]).squeeze())
            c_char_var = make_char_vector(cc, w2i_c, ctx_sent_maxlen, ctx_word_maxlen)
            c_word_var = make_word_vector(c, w2i_w, ctx_sent_maxlen)
            q_char_var = make_char_vector(qc, w2i_c, query_sent_maxlen, query_word_maxlen)
            q_word_var = make_word_vector(q, w2i_w, query_sent_maxlen)
            p1, p2 = model(c_char_var, c_word_var, q_char_var, q_word_var)
            loss_p1 = nn.NLLLoss()(p1, a_beg)
            # loss_p2 = nn.NLLLoss()(p2, a_end)
            if i % (args.batch_size*20) == 0:
                now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                # print('[{}] {:.1f}%, loss_p1: {:.3f}, loss_p2: {:.3f}'.format(now, 100*i/len(data), loss_p1.data[0], loss_p2.data[0]))
                print('[{}] Epoch {} {:.1f}%, loss_p1: {:.3f}'.format(now, epoch, 100*i/len(data), loss_p1.data[0]))
                print('before param 0', list(model.parameters())[0][0][:5])
                # test(model)
                # p1_rank, p2_rank = batch_ranking(p1, p2)
                p1_rank, p2_rank = batch_ranking(p1, p1)
                for rank in range(1): # N-best, currently 1-best
                    p1_rank_id = p1_rank[0][rank]
                    p2_rank_id = p2_rank[0][rank]
                    print('Rank {}, p1_result={}, p2_result={}'.format(
                        rank+1, p1_rank_id == a_beg.data[0], p2_rank_id == a_end.data[0]))
                # TODO calc acc, save every epoch wrt acc

            model.zero_grad()
            # (loss_p1+loss_p2).backward()
            loss_p1.backward()
            # print('before param 0', list(model.parameters())[0][0][:5])
            optimizer.step()
            # print('after param 0', list(model.parameters())[0][0][:5])

        # end eopch
        filename = '{}/Epoch-{}.model'.format('./checkpoints', epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, True, filename=filename)


# test() {{{
def test(model, batch_size=args.batch_size+2):
    model.eval()
    p1_acc_count = 0
    p2_acc_count = 0
    for i in tqdm(range(0, len(dev_data)-batch_size, batch_size)): # TODO last elms
        batch_data = dev_data[i:i+batch_size]
        c = [d[1] for d in batch_data]
        cc = [d[2] for d in batch_data]
        q = [d[3] for d in batch_data]
        qc = [d[4] for d in batch_data]
        a_beg = to_var(torch.LongTensor([d[5][0] for d in batch_data]).squeeze()) # TODO: multi target
        a_end = to_var(torch.LongTensor([d[5][1] for d in batch_data]).squeeze())
        c_char_var = make_char_vector(cc, w2i_c, ctx_sent_maxlen, ctx_word_maxlen)
        c_word_var = make_word_vector(c, w2i_w, ctx_sent_maxlen)
        q_char_var = make_char_vector(qc, w2i_c, query_sent_maxlen, query_word_maxlen)
        q_word_var = make_word_vector(q, w2i_w, query_sent_maxlen)
        p1, p2 = model(c_char_var, c_word_var, q_char_var, q_word_var)
        p1_rank, p2_rank = batch_ranking(p1, p1)

        for n, (pp1, pp2) in enumerate(zip(p1, p1)):
            # if i % 100 == 0:
            # now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            for rank in range(1): # N-best, currently 1-best
                p1_rank_id = p1_rank[n][rank]
                p2_rank_id = p2_rank[n][rank]
                # print('Rank {}, p1_result={}, p2_result={}'.format(
                #     rank+1, p1_rank_id==a_beg.data[n], p2_rank_id==a_end.data[n]))
                if p1_rank_id == a_beg.data[n]:
                    p1_acc_count += 1
                if p2_rank_id == a_end.data[n]:
                    p2_acc_count += 1
    N = len(dev_data)
    print('======== Test result ========')
    print('p1 acc: {:.3f}, p2 acc: {:.3f}'.format(p1_acc_count/N, p2_acc_count/N))
# }}}


model = AttentionNet(args)
# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5, weight_decay=0.999)
# optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
if torch.cuda.is_available():
    print('use cuda')
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())

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
# print('parameters-----')
# for parameter in model.parameters():
#     print(parameter.size())

if args.test_mode == 1:
    test(model)
else:
    train(model, optimizer)
print('finish')
