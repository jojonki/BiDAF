import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# In : (N, sentence_len, word_len, vocab_size_c)
# Out: (N, sentence_len, embd_size)
class CharEmbedding(nn.Module):
    def __init__(self, args):
        super(CharEmbedding, self).__init__()
        self.embd_size = args.embd_size
        self.embedding = nn.Embedding(args.vocab_size_c, args.embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv = nn.ModuleList([nn.Conv2d(1, args.out_chs, (f[0], f[1])) for f in args.filters])
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(args.out_chs*len(args.filters), 1)
        
    def forward(self, x):
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        bs = x.size(0)
        seq_len = x.size(1)
        word_len = x.size(2)
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        x = self.embedding(x) # (N*seq_len, word_len, embd_size)
        x = x.view(*input_shape, -1) # (N, seq_len, word_len, embd_size)
        x = x.sum(2) # (N, seq_len, embd_size)
        
        return x
