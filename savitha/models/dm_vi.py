import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

class DiffusionBlockVI(nn.Module):
    def __init__(self, hidden_layer, dropout_prob=0.3):
        super(DiffusionBlockVI, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_layer)
        self.linear1 = BayesianLinear(hidden_layer, hidden_layer)
        self.linear2 = BayesianLinear(hidden_layer, hidden_layer)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x


class DiffusionModelVI(nn.Module):
    def __init__(self, nfeatures, nblocks=12, hidden_layer=512, dropout_prob=0.1):
        super(DiffusionModelVI, self).__init__()
        self.time_embed_dim = hidden_layer
        self.time_embed = nn.Sequential(
            BayesianLinear(1, hidden_layer),
            nn.SiLU(),
            BayesianLinear(hidden_layer, hidden_layer)
        )
        
        self.inblock = BayesianLinear(nfeatures + hidden_layer, hidden_layer)
        self.midblocks = nn.ModuleList([DiffusionBlockVI(hidden_layer, dropout_prob) for _ in range(nblocks)])
        self.final_norm = nn.LayerNorm(hidden_layer)
        self.outblock = BayesianLinear(hidden_layer, nfeatures)
        
    def forward(self, x, t):
        t = t.to(torch.float32)
        t_emb = self.time_embed(t)
        val = torch.cat([x, t_emb], dim=-1)
        val = self.inblock(val)
        for block in self.midblocks:
            val = block(val)
        val = self.final_norm(val)
        val = self.outblock(val)
        return val
