from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from rl.policies.utils import MLP
from rl.policies.actor_critic import Actor, Critic


class MlpActor(Actor):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__(config, ob_space, ac_space, tanh_policy)

        self._ac_space = ac_space # e.g. ActionSpace(shape=OrderedDict([('default', 8)]),minimum=-1.0, maximum=1.0)

        # observation # e.g. OrderedDict([('object-state', [10]), ('robot-state', [36])])
        input_dim = sum([np.prod(x) for x in ob_space.values()]) # [[226,226,3],[36],[10]] (226*226*3)+36+10

        # build sequential layers of neural network
        self.fc = MLP(config, input_dim, config.rl_hid_size, [config.rl_hid_size]) # inp= 46, output= 64
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, size in ac_space.shape.items(): # shape=OrderedDict([('default', 8)])
            self.fc_means.update({k: MLP(config, config.rl_hid_size, size)}) # MLP here defines the out_layer of nn
            self.fc_log_stds.update({k: MLP(config, config.rl_hid_size, size)})

    def forward(self, ob): # extracts input values from ob odict and passes it to nn
        inp = list(ob.values()) # [tensor([-0.007,.....,-0.51]), tensor([-0.08,.....,0.08])]
        if len(inp[0].shape) == 1: #[36]
            inp = [x.unsqueeze(0) for x in inp] # change the tensor shape from e.g. [36] to [1, 36]

        # concatenates the tensor to [1, 46], passes to model and activation_fn; out is a tensor of shape (1, hid_size)
        out = self._activation_fn(self.fc(torch.cat(inp, dim=-1)))
        out = torch.reshape(out, (out.shape[0], -1)) # [1, 64]

        means, stds = OrderedDict(), OrderedDict()
        for k in self._ac_space.keys():
            mean = self.fc_means[k](out) # passes previous layers output to last layer to produce action tensor [1,8]
            log_std = self.fc_log_stds[k](out) # passes previous layers output to last layer for s.d. tensor
            log_std = torch.clamp(log_std, -10, 2) # clips between min and max
            std = torch.exp(log_std.double()) # exponential on s.d.
            means[k] = mean
            stds[k] = std

        return means, stds


class MlpCritic(Critic):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__(config)

        input_dim = sum([np.prod(x) for x in ob_space.values()])
        if ac_space is not None:
            input_dim += ac_space.size

        self.fc = MLP(config, input_dim, 1, [config.rl_hid_size] * 2)

    def forward(self, ob, ac=None):
        inp = list(ob.values())
        if len(inp[0].shape) == 1:
            inp = [x.unsqueeze(0) for x in inp]

        if ac is not None:
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            inp.extend(ac)

        out = self.fc(torch.cat(inp, dim=-1))
        out = torch.reshape(out, (out.shape[0], 1))

        return out
