import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# In : (N, sentence_len, word_len, vocab_size_c)
# Out: (N, sentence_len, embd_size)
class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, embd_size, out_chs, filters):
        super(CharEmbedding, self).__init__()
        self.embd_size = embd_size
        self.embedding = nn.Embedding(vocab_size, embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv = nn.ModuleList([nn.Conv2d(1, out_chs, (f[0], f[1])) for f in filters])
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(out_chs*len(filters), 1)
        
    def forward(self, x):
        print('x', x.size()) # (N, seq_len, word_len)
        bs = x.size(0)
        seq_len = x.size(1)
        word_len = x.size(2)
        embd = Variable(torch.zeros(bs, seq_len, self.embd_size))
        for i, elm in enumerate(x): # every sample
            for j, chars in enumerate(elm): # every sentence. [ [‘w’, ‘h’, ‘o’, 0], [‘i’, ‘s’, 0, 0], [‘t’, ‘h’, ‘i’, ‘s’] ]
                chars_embd = self.embedding(chars.unsqueeze(0)) # (N, word_len, embd_size) [‘w’,‘h’,‘o’,0]
                chars_embd = torch.sum(chars_embd, 1) # (N, embd_size). sum each char's embedding
                embd[i,j] = chars_embd[0] # set char_embd as word-like embedding

        x = embd # (N, seq_len, embd_dim)
        x = embd.unsqueeze(1) # (N, Cin, seq_len, embd_dim), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout) 
        x = [F.relu(conv(x)) for conv in self.conv] # (N, Cout, seq_len, embd_dim-filter_w+1). stride == 1
        
        # [(N,Cout,Hout,Wout) -> [(N,Cout,Hout*Wout)] * len(filter_heights)
        # [(N, seq_len, embd_dim-filter_w+1, Cout)] * len(filter_heights)
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]
        
        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [torch.sum(xx, 2) for xx in x]
        out = torch.cat(x, 1)
        return out
