import numpy as np
import torch
import torch.nn.functional as F

import nflows.transforms.base as base

class Dequantization(base.Transform):
    """
        Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
    """
    def __init__(self, alpha=1e-5, quants=256):
        super().__init__()

        self.alpha = alpha
        self.quants = quants
        
    def inverse(self, inputs, context=None):
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        inputs, ldj = self.sigmoid(inputs, ldj)
        inputs = inputs * self.quants
        ldj += np.log(self.quants) * np.prod(inputs.shape[1:])
        inputs = torch.floor(inputs).clamp(min=0, max=self.quants-1).to(torch.int32)
        return inputs, ldj

    def forward(self, inputs, context=None):
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)
        inputs, ldj = self.dequant(inputs, ldj)
        inputs, ldj = self.logit(inputs, ldj)
        return inputs, ldj

    def logit(self, z, ldj):
        z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
        ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
        ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
        z = torch.log(z) - torch.log(1-z)
        
        return z, ldj

    def sigmoid(self, z, ldj):
        ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
        z = torch.sigmoid(z)
            
        return z, ldj
    
    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z)
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj

