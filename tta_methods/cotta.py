from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import utils.cotta_utils.cotta_transforms as my_transforms
from time import time
from tta_methods import Basic_Wrapper


class CoTTA(Basic_Wrapper):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, config):#, optimizer, steps=1, episodic=False):
        super().__init__(model, config)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = self.copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()    
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                std = [ 1., 1., 1. ])
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        self.model_ema.train()
        # Teacher Prediction
        
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        to_aug = anchor_prob.mean(0)<0.1
        if to_aug: 
            for i in range(N):
                # x=self.denormalize(x)
                outputs_  = self.model_ema(x=self.normalize(self.transform(self.denormalize(x)))).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) 
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.001).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema
    
    def prepare_model_and_optimizer(self, model, config):
        model = self.configure_model(model, config.model.update_bn_only)
        params, _ = self.collect_params(model)
        # LR and Momentum Optimal for SHOT and SHOT-IM harcoded here
        optimizer = torch.optim.SGD(params, 
                                    lr=config.optimization.lr, 
                                    momentum=config.optimization.momentum,
                                    dampening=config.optimization.dampening,
                                    weight_decay=config.optimization.weight_decay,
                                    nesterov=config.optimization.nesterov)
        return model, optimizer

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        model_anchor = deepcopy(model)
        optimizer_state = deepcopy(optimizer.state_dict())
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor
    
def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5
    
    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)