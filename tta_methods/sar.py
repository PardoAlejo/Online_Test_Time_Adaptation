"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from utils.sar_utils.sam import SAM
from tta_methods import Basic_Wrapper


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(Basic_Wrapper):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, config):
        super().__init__(model, config)

        self.margin_e0 = math.log(config.dataset.num_classes)*config.model.margin_e_constant  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = config.model.reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        
        
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, reset_flag = self.forward_and_adapt(x)
            if reset_flag:
                self.reset()

        return outputs
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        # forward
        outputs = self.model(x)
        # adapt
        # filtering reliable samples/gradients for further adaptation; first time forward
        entropys = softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(self.model(x))
        entropys2 = entropys2[filter_ids_1]  # second time forward  
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item())  # record moving average loss values for model recovery

        # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        # perform model recovery
        reset_flag = False
        if self.ema is not None:
            if self.ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        return outputs, reset_flag

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.ema = None

    def collect_params(self, model, update_bn_only=None):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    
    def configure_model(self, model, update_bn_only=True):
        """Configure model for use with SAR."""
        # train mode, because SAR optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what SAR updates
        model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model
    
    def prepare_model_and_optimizer(self, model, config):
        model = self.configure_model(model)
        params, param_names = self.collect_params(model)
        base_optimizer = torch.optim.SGD
        lr = (0.00025 / 64) * config.dataset.batch_size * 2 if config.dataset.batch_size < 32 else 0.00025
        if config.dataset.batch_size == 1:
            lr = 2*lr
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=config.optimization.momentum)
        return model, optimizer

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
