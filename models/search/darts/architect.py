import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from .detector import DetBenchTrainImagePair

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args, optimizer, device):
        self.network_weight_decay = args.weight_decay
        self.arch_training_bench = DetBenchTrainImagePair(model, create_labeler=True)
        self.arch_training_bench = self.arch_training_bench.to(device)
        self.optimizer = optimizer
    
    def log_learning_rate(self, logger):
        for param_group in self.optimizer.param_groups:
            logger.info("Architecture Learning Rate: {}".format(param_group['lr']))
            break
    
    def step(self, thermal_img_tensor, rgb_img_tensor, target):
        self.optimizer.zero_grad()
        self._backward_step(thermal_img_tensor, rgb_img_tensor, target)
        self.optimizer.step()

    def _backward_step(self, thermal_img_tensor, rgb_img_tensor, target):
        output = self.arch_training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False)
        loss = output['loss']
        loss.backward()
    