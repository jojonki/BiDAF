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
# from layers.char_embedding import CharEmbedding
from layers.char_embedding import CharEmbedding
from layers.word_embedding import WordEmbedding
from layers.highway import Highway

class AttentionNet(nn.Module):
    def __init__(self, args):
        super(AttentionNet, self).__init__()
        self.embd_size = args.w_embd_size
        self.d = self.embd_size * 2 # word_embedding + char_embedding
        self.ans_size = args.ans_size

        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(self.embd_size)
        self.ctx_embd_layer = nn.GRU(self.d, self.d, bidirectional=True, dropout=0.2)

        self.W = nn.Parameter(torch.rand(1, 6*self.d, 1).type(torch.FloatTensor), requires_grad=True) # (N, 6d, 1) for bmm (N, T*J, 6d)
        self.modeling_layer = nn.GRU(8*self.d, self.d, bidirectional=True, dropout=0.2)
        self.p1_layer = nn.Linear(10*self.d, args.ans_size)
        self.p2_lstm_layer = nn.GRU(2*self.d, 2*self.d, bidirectional=True, dropout=0.2)
        self.p2_layer = nn.Linear(12*self.d, args.ans_size)
        
    def build_contextual_embd(self, x_c, x_w):
        # 1. Caracter Embedding Layer
        char_embd = self.char_embd_net(x_c) # (N, seq_len, embd_size)
        if torch.cuda.is_available():
            char_embd = char_embd.cuda()
        # 2. Word Embedding Layer
        word_embd = self.word_embd_net(x_w) # (N, seq_len, embd_size)
        if torch.cuda.is_available():
            word_embd = word_embd.cuda()
        # Highway Networks for 1. and 2.
        char_embd = self.highway_net(char_embd)
        word_embd = self.highway_net(word_embd)
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, d==embd_size*2)
        
        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out
        
    def forward(self, ctx_c, ctx_w, query_c, query_w):
        batch_size = ctx_c.size(0)
        T = ctx_w.size(1)   # context sentence length (word level)
        J = query_w.size(1) # query sentence length   (word level)
        
        # 1. Caracter Embedding Layer 
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.build_contextual_embd(ctx_c, ctx_w)     # (N, T, 2d)
        embd_query   = self.build_contextual_embd(query_c, query_w) # (N, J, 2d)
        
        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, T, J, 2*self.d)            # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)     # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)     # (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
        cat_data = cat_data.view(batch_size, -1, 6*self.d) # (N, T*J, 6d)
        S = torch.bmm(cat_data, self.W.expand(batch_size, 6*self.d, 1)) # (N, T*J, 1)
        S = S.view(batch_size, T, J) # (N, T, J), unsqueeze last dim
        S = S.view(batch_size*T, J)
        S = torch.stack([F.softmax(S[i]) for i in range(len(S))], 0) # softmax for each row
        S = S.view(batch_size, T, J) # (N, T, J), unsqueeze last dim

        # Context2Query
        c2q = torch.bmm(S, embd_query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        tmp_b = torch.max(S, 2)[0] # (N, T)
        b = torch.stack([F.softmax(tmp_b[i]) for i in range(batch_size)], 0) # (N, T), softmax for each row
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times
        
        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 8d)
        
        # 5. Modeling Layer
        M, _ = self.modeling_layer(G) # M: (N, T, 2d)
        
        # 5. Output Layer
        G_M = torch.cat((G, M), 2) # (N, T, 10d)
        G_M = G_M.sum(1) #(N, 10d)
        p1 = F.log_softmax(self.p1_layer(G_M)) # (N, T)
        
        M2, _ = self.p2_lstm_layer(M) # (N, T, 4d)
        G_M2 = torch.cat((G, M2), 2) # (N, T, 12d)
        G_M2 = G_M2.sum(1) # (N, 12d)(N, T)
        p2 = F.log_softmax(self.p2_layer(G_M2)) # (N, T)
        
        return p1, p2
