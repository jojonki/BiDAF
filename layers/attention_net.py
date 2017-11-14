import numpy as np
import os
import json
from nltk.tokenize import word_tokenize

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

class AttentionNet(nn.Module):
    def __init__(self, args):
        super(AttentionNet, self).__init__()
        self.embd_size = args.embd_size
        self.ans_size = args.ans_size
        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(args.embd_size*2)# TODO share is ok?
        self.ctx_embd_layer = nn.GRU(args.embd_size*2, args.embd_size*2, bidirectional=True)
        self.W = nn.Parameter(torch.rand(3*2*2* args.embd_size, 1).type(torch.FloatTensor), requires_grad=True)
#         self.beta = nn.Parameter(torch.rand(8*2*2* args.embd_size).type(torch.FloatTensor).view(1, -1), requires_grad=True)
        self.modeling_layer = nn.GRU(args.embd_size*2*8, args.embd_size*2, bidirectional=True)
        self.p1_layer = nn.Linear(args.embd_size*2*10, args.ans_size)
        self.p2_lstm_layer = nn.GRU(args.embd_size*2*2, args.embd_size*2*2, bidirectional=True)
        self.p2_layer = nn.Linear(args.embd_size*2*12, args.ans_size)
        
    def build_contextual_embd(self, x_c, x_w):
        # 1. Caracter Embedding Layer
        char_embd = self.char_embd_net(x_c) # (N, seq_len, embd_size)
        if torch.cuda.is_available():
            char_embd = char_embd.cuda()
        # 2. Word Embedding Layer
        word_embd = self.word_embd_net(x_w) # (N, seq_len, embd_size)
        if torch.cuda.is_available():
            word_embd = word_embd.cuda()
        # Highway Networks of 1. and 2.
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, embd_size*2)
        embd = self.highway_net(embd)
        
        # 3. Contextual  Embedding Layer
        ctx_embd_out, ctx_embd_h = self.ctx_embd_layer(embd)
        return ctx_embd_out
        
    def forward(self, ctx_c, ctx_w, query_c, query_w):
        batch_size = ctx_c.size(0)
        
        # 1. Caracter Embedding Layer 
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.build_contextual_embd(ctx_c, ctx_w) # (N, T, 2d)
        ctx_len = embd_context.size(1)
        embd_query   = self.build_contextual_embd(query_c, query_w) # (N, J, 2d)
        query_len = embd_query.size(1)
        
        # 4. Attention Flow Layer
        # Context2Query
        shape = (batch_size, ctx_len, query_len, self.embd_size*2*2) # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2) # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape)
        embd_query_ex = embd_query.unsqueeze(1) # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d)
        cat_data = cat_data.view(batch_size, -1, 6*2*self.embd_size)
        S = torch.bmm(cat_data, self.W.unsqueeze(0).expand(batch_size, 6*2*self.embd_size, 1))
        S = S.view(batch_size, ctx_len, query_len)
        
        c2q = torch.bmm(S, embd_query) # (N, T, 2d)
        # Query2Context
        tmp_b = torch.max(S, 2)[0]
        b = torch.stack([F.softmax(tmp_b[i]) for i in range(batch_size)], 0) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context).squeeze() # (N, 2d)
        q2c = q2c.unsqueeze(1) # (N, 1, 2d)
        q2c = q2c.repeat(1, ctx_len, 1) # (N, T, 2d)
        
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 8d)
        
        # 5. Modeling Layer
        M, _ = self.modeling_layer(G) # M: (N, T, 2d)
        
        # 5. Output Layer
        G_M = torch.cat((G, M), 2) # (N, T, 10d)
        G_M = G_M.sum(1) #(N, 10d)
        p1 = F.softmax(self.p1_layer(G_M)) # (N, T)
        
        M2, _ = self.p2_lstm_layer(M) # (N, T, 4d)
        G_M2 = torch.cat((G, M2), 2) # (N, T, 12d)
        G_M2 = G_M2.sum(1) # (N, 12d)(N, T)
        p2 = F.softmax(self.p2_layer(G_M2)) # (N, T)
        
        return p1, p2
