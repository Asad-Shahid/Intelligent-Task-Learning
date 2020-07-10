import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(tensor): # Initialization
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound) # fills a tensor with unifrom distribution between bounds

class MLP(nn.Module):
    def __init__(self, config, input_dim, output_dim, hid_dims=[]):
        super().__init__()
        #activation_fn = getattr(F, config.activation)
        activation_fn = nn.ReLU()

        fc = []
        prev_dim = input_dim
        for d in hid_dims:
            fc.append(nn.Linear(prev_dim, d))
            fanin_init(fc[-1].weight)
            fc[-1].bias.data.fill_(0.1)
            fc.append(activation_fn)
            prev_dim = d
        fc.append(nn.Linear(prev_dim, output_dim))
        fc[-1].weight.data.uniform_(-1e-3, 1e-3)
        fc[-1].bias.data.uniform_(-1e-3, 1e-3)
        self.fc = nn.Sequential(*fc)

    def forward(self, ob):
        return self.fc(ob)
