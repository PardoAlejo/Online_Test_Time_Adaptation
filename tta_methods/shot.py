from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from tta_methods import Basic_Wrapper

#rewrite SHOTIM using the Basic_Wrapper
class SHOT(Basic_Wrapper):
    """
    SHOT method from https://arxiv.org/abs/2002.08546
    """
    def __init__(self, model, config):
        super().__init__(model, config)
        self.beta_clustering_loss = config.model.beta_clusteting_loss
    
    def loss_fn(self, outputs):
            
        softmax_out = outputs.softmax(1)
        msoftmax = softmax_out.mean(0)

        # SHOT-IM
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
        
        return loss

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        
        outputs, features = self.model(x, return_feature=True)
        loss = self.loss_fn(outputs, features)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs
    
    def loss_fn(self, outputs, features):
        device = outputs.device
        softmax_out = outputs.softmax(1)
        msoftmax = softmax_out.mean(0)

        # ================ SHOT-IM ================
        l_ent = - (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
        l_div =  (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
        
        loss = l_ent + l_div
        # ================ SHOT-IM ================
        
        
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        # normalize features
        features = features / features.norm(dim=1, keepdim=True)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L386
        # features = (features.t()/torch.norm(features,p=2,dim=1)).t()
        
        # Compute clustering loss
        # Compute centroids of each class            
        K = outputs.shape[1]
        aff = softmax_out.to(device)
        
        initial_centroids = torch.matmul(aff.t(), features)
        # Equivalent to the following line in the original implementation
        # https://github.com/tim-learn/SHOT/blob/07d0c713e4882e83fded1aff2a447dff77856d64/digit/uda_digit.py#L391
        # initial_centroids = aff.transpose().dot(features)
        
        #normalize centroids
        initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
        # aff.sum(0, keepdim=True).t() is equivalente to aff.sum(0)[:, None]

        # Compute distances to centroids
        distances = torch.cdist(features, initial_centroids, p=2)
        # Compute pseudo labels
        pseudo_labels = distances.argmin(axis=1).to(device)
        
        # I don't know why they do this, but it's in the original implementation
        for _ in range(1):
            aff = torch.eye(K, device=device)[pseudo_labels]
            initial_centroids = torch.matmul(aff.t(), features)
            initial_centroids = initial_centroids / (1e-8 + aff.sum(0, keepdim=True).t())
            distances = torch.cdist(features, initial_centroids, p=2)
            pseudo_labels = distances.argmin(axis=1)
            
        # Compute clustering loss
        loss += self.beta_clustering_loss * F.cross_entropy(outputs, pseudo_labels)
        # =============== Full SHOT, SHOT-IM + clustering loss ===============
        
        return loss