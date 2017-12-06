import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.char_embedding import CharEmbedding
from layers.word_embedding import WordEmbedding
from layers.highway import Highway


class AttentionNet(nn.Module):
    def __init__(self, args):
        super(AttentionNet, self).__init__()
        self.embd_size = args.w_embd_size
        self.d = self.embd_size * 2 # word_embedding + char_embedding
        # self.d = self.embd_size # only word_embedding

        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(self.d)
        self.ctx_embd_layer = nn.LSTM(self.d, self.d, bidirectional=True, dropout=0.2)

        self.W = nn.Linear(6*self.d, 1, bias=False)

        self.modeling_layer = nn.LSTM(8*self.d, self.d, num_layers=2, bidirectional=True, dropout=0.2)

        self.p1_layer = nn.Linear(10*self.d, 1, bias=False)
        self.p2_lstm_layer = nn.LSTM(2*self.d, self.d, bidirectional=True, dropout=0.2)
        self.p2_layer = nn.Linear(10*self.d, 1)

    def build_contextual_embd(self, x_c, x_w):
        # 1. Caracter Embedding Layer
        char_embd = self.char_embd_net(x_c) # (N, seq_len, embd_size)
        # 2. Word Embedding Layer
        word_embd = self.word_embd_net(x_w) # (N, seq_len, embd_size)
        # Highway Networks for 1. and 2.
        embd = torch.cat((char_embd, word_embd), 2) # (N, seq_len, d=embd_size*2)
        embd = self.highway_net(embd) # (N, seq_len, d=embd_size*2)

        # 3. Contextual  Embedding Layer
        ctx_embd_out, _h = self.ctx_embd_layer(embd)
        return ctx_embd_out

    def forward(self, ctx_w, ctx_c, query_w, query_c):
        batch_size = ctx_w.size(0)
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
        S = self.W(cat_data).squeeze() # (N, T, J)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2) # (N, T, 8d)

        # 5. Modeling Layer
        M, _h = self.modeling_layer(G) # M: (N, T, 2d)

        # 6. Output Layer
        G_M = torch.cat((G, M), 2) # (N, T, 10d)
        p1 = F.softmax(self.p1_layer(G_M).squeeze(), dim=-1) # (N, T)

        M2, _ = self.p2_lstm_layer(M) # (N, T, 2d)
        G_M2 = torch.cat((G, M2), 2) # (N, T, 10d)
        p2 = F.softmax(self.p2_layer(G_M2).squeeze(), dim=-1) # (N, T)

        return p1, p2
