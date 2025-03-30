import torch
import torch.nn as nn


class DiffusionBlock(nn.Module):
    def __init__(self, hidden_layer, dropout_prob=0.1):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(hidden_layer, hidden_layer)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, nfeatures: int, nblocks: int = 2, hidden_layer = 64, dropout_prob=0.1):
        super(DiffusionModel, self).__init__()

        self.inblock = nn.Linear(nfeatures+1, hidden_layer)
        self.midblocks = nn.ModuleList([DiffusionBlock(hidden_layer) for _ in range(nblocks)])
        self.dropout = nn.Dropout(dropout_prob)
        self.outblock = nn.Linear(hidden_layer, nfeatures)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        val = torch.hstack([x, t])
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.dropout(val)
        val = self.outblock(val)
        return val
    