import torch
import torch.nn as nn

import config

class LoRA_Parameterization(nn.Module):
    def __init__(self, d, k, rank=config.RANK, alpha=config.ALPHA):
        super().__init__()
        self.LoRA_Matrix_A = nn.Parameter(torch.zeros((rank, k)).to(config.DEVICE))
        self.LoRA_Matrix_B = nn.Parameter(torch.zeros((d, rank)).to(config.DEVICE))

        nn.init.normal_(self.LoRA_Matrix_A, mean=0, std=1)

        self.scale = alpha / rank
        self.enabled = True
    
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + \
                   torch.matmul(self.LoRA_Matrix_B, self.LoRA_Matrix_A) \
                    .view(original_weights.shape) * self.scale
        else:
            return original_weights
