import torch
import torch.nn as nn
import torch.nn.functional as F

# In : (N, sentence_len, vocab_size_w)
# Out: (N, sentence_len, embd_size)
class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embd_size, is_train_embd=False, pre_embd_w=None):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd_w is not None:
            self.embedding.weight = nn.Parameter(pre_embd_w, requires_grad=is_train_embd)
        
    def forward(self, x):
        x = self.embedding(x)
        out = F.relu(x)
        print('out', out.size())

        return out
