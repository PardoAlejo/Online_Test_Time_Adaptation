"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""

from copy import deepcopy
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from tta_methods import Basic_Wrapper

from utils.dataloader import ImageNet_val_subset_data
import math

# This code is adapted from the official implementation:
# https://github.com/mr-eggplant/EATA

class EATA(Basic_Wrapper):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, config):
        super().__init__(model, config)
        
        self.fisher_alpha = config.model.fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = math.log(config.dataset.num_classes)*config.model.margin_e_constant # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = config.model.d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    
    def prepare_model_and_optimizer(self, model, config):
        model = self.configure_model(model)
        params, _ = self.collect_params(model)

        if config.model.method == 'eta':
            self.fishers = None
        elif config.model.method == 'eata':
            self.fishers = compute_fishers(model, params, config)

        optimizer = torch.optim.SGD(params, lr=config.optimization.lr, momentum=config.optimization.momentum)
        
        return model, optimizer
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average  probability vector over all previous samples
        """
        # forward
        outputs = self.model(x)
        # adapt
        entropys = softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0]>-0.1)
        entropys = entropys[filter_ids_1] 
        # filter redundant samples
        if self.current_model_probs is not None: 
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1])**2).sum()
            loss += ewc_loss
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def compute_fishers(model, params, config):

    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    fisher_loader = ImageNet_val_subset_data(data_dir=config.dataset.paths.imagenet, 
                                            batch_size=config.dataset.batch_size, 
                                            shuffle=config.dataset.shuffle, 
                                            subset_size=config.model.fisher_size)
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    
    print("Computing fishers ...")

    for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
        images = images.cuda(config.run.gpu, non_blocking=True)
        targets = targets.cuda(config.run.gpu, non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()

    print("Computing fisher matrices finished")
    
    del ewc_optimizer

    return fishers

    

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)