import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, in_size, n_layers=2, act=F.relu, gate_act=F.sigmoid):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.act = act
        self.gate_act = gate_act

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

        self.gate = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = self.act(self.normal_layer[i](x))
            gate = self.gate_act(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x

