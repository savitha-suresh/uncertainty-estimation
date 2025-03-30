import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionBlockV2(nn.Module):
    def __init__(self, hidden_layer, dropout_prob=0.1):
        super(DiffusionBlockV2, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_layer)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = F.silu(x)  # SiLU/Swish activation
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + x  # Residual connection

class DiffusionModelV2(nn.Module):
    def __init__(self, nfeatures, nblocks=12, hidden_layer=512, dropout_prob=0.1):
        super(DiffusionModelV2, self).__init__()
        self.time_embed_dim = hidden_layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_layer),
            nn.SiLU(),
            nn.Linear(hidden_layer, hidden_layer)
        )
        
        self.inblock = nn.Linear(nfeatures + hidden_layer, hidden_layer)
        self.midblocks = nn.ModuleList([DiffusionBlockV2(hidden_layer, dropout_prob) for _ in range(nblocks)])
        self.final_norm = nn.LayerNorm(hidden_layer)
        self.outblock = nn.Linear(hidden_layer, nfeatures)
        
    def forward(self, x, t):
        # Better time embedding
        t = t.to(torch.float32)
        t_emb = self.time_embed(t)
        
        # Concatenate features with time embedding
        val = torch.cat([x, t_emb], dim=-1)
        val = self.inblock(val)
        
        # Process through blocks
        for block in self.midblocks:
            val = block(val)
            
        val = self.final_norm(val)
        val = self.outblock(val)
        return val