from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from tta_methods import Basic_Wrapper

#rewrite SHOTIM using the Basic_Wrapper
class SHOTIM(Basic_Wrapper):
    """
    SHOTIM method from https://arxiv.org/abs/2002.08546
    """
    def loss_fn(self, outputs):
            
        softmax_out = outputs.softmax(1)
        msoftmax = softmax_out.mean(0)

        # SHOT-IM
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
        
        return loss