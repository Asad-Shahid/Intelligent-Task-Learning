from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.distributions


# Normal
FixedNormal = torch.distributions.Normal

normal_init = FixedNormal.__init__
FixedNormal.__init__ = lambda self, mean, std: normal_init(self, mean.double(), std.double())

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions.double()).sum(-1, keepdim=True).float()

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1).float()

FixedNormal.mode = lambda self: self.mean.float()

normal_sample = FixedNormal.sample
FixedNormal.sample = lambda self: normal_sample(self).float()

normal_rsample = FixedNormal.rsample
FixedNormal.rsample = lambda self: normal_rsample(self).float()

class MixedDistribution(nn.Module):
    def __init__(self, distributions):
        super().__init__()
        assert isinstance(distributions, OrderedDict)
        self.distributions = distributions

    def mode(self):
        return OrderedDict([(k, dist.mode()) for k, dist in self.distributions.items()])

    def sample(self):
        return OrderedDict([(k, dist.sample()) for k, dist in self.distributions.items()])

    def rsample(self):
        return OrderedDict([(k, dist.rsample()) for k, dist in self.distributions.items()])

    def log_probs(self, x):
        assert isinstance(x, dict)
        return OrderedDict([(k, dist.log_probs(x[k])) for k, dist in self.distributions.items()])

    def entropy(self):
        return sum([dist.entropy() for dist in self.distributions.values()])

