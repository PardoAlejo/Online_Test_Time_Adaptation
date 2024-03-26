
import torch.nn as nn
from tta_methods import Basic_Wrapper
# This code is adapted from the paper:
# https://arxiv.org/pdf/1603.04779.pdf
class AdaBn(Basic_Wrapper):
    """BN Adapts the model by updating the statistics of the BatchNorm Layers.
    """
    def forward_and_adapt(self, x):
        return self.model(x)