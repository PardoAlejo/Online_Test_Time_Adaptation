from copy import deepcopy

import torch
import torch.nn as nn

class Basic_Wrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model, self.optimizer = self.prepare_model_and_optimizer(model, config)
        
        self.episodic = config.run.episodic
        self.steps = config.optimization.steps if hasattr(config.optimization, 'steps') else 1
        
        if self.episodic:
            self.copy_model_and_optimizer()

    def prepare_model_and_optimizer(self, model, config):
        model = self.configure_model(model, config.model.update_bn_only)
        params, _ = self.collect_params(model, config.model.update_bn_only)
        # LR and Momentum Optimal for SHOT and SHOT-IM harcoded here
        optimizer = torch.optim.SGD(params, 
                                    lr=config.optimization.lr, 
                                    momentum=config.optimization.momentum)
        return model, optimizer
    
    
    def configure_model(self, model, update_bn_only=True):
        """Configure model for use with tent."""
        # train mode, because shot optimizes the model to minimize entropy
        model.train()
        if update_bn_only:
            # In case we want to update only the BN layers
            model.requires_grad_(False)
            # configure norm for updates: enable grad + force batch statisics
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None      
        else:
            # is this needed? review later
            model.requires_grad_(True)
            # Freeze FC layers
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.requires_grad_(False)
        return model
    
    def collect_params(self, model, update_bn_only=True):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        if update_bn_only:
            for nm, m in model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
        else:
            for nm, m in model.named_modules():
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    
    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
    
    def forward(self, x):
        if self.episodic:
            self.reset()
            
        for i in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs
    
    # Implement loss_fn for the method
    # Dummy loss function
    def loss_fn(self, outputs):
        return outputs.mean()
    
    # Example of forward and adapt function
    # Probably changes for every method
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        outputs = self.model(x)
        loss = self.loss_fn(outputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs
    
    def reset(self):
        self.load_model_and_optimizer()