# Alternate Model for LightGCN
# Paper suggested it had worse performance, tested to confirm.

import torch
import torch.nn as nn

class LightGCNLayer(nn.Module):
    def __init__(self, d_model):
        super(LightGCNLayer, self).__init__()
        self.linear = nn.Identity()

    def forward(self, x, adj):
        return torch.matmul(adj, x)