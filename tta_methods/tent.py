from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from tta_methods import Basic_Wrapper

# This code is adapted from the official implementation:
# https://github.com/DequanWang/tent/blob/master/tent.py
# steps=1, lr=0.00025, momentum=0.9 for ImageNet
class Tent(Basic_Wrapper):
    """
    Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def loss_fn(self, outputs):
        return -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean(0)
