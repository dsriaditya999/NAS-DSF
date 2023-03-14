import torch
import torch.nn as nn
import torch.nn.functional as F

from .genotypes import *

OPS = {
    'none': lambda args: Zero(),
    'skip': lambda args: Identity()
}

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        out = x.mul(0.)
        return out



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FusionMixedOp(nn.Module):

    def __init__(self, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](args)
            self._ops.append(op)

    def forward(self, x, weights):
        out = sum(w * op(x) for w, op in zip(weights, self._ops))
        return out