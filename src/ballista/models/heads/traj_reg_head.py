import torch
import torch.nn as nn
from ballista.models.registry import HEADS


@HEADS.register("traj_reg_head")
class TrajRegHead(nn.Module):
    def __init__(self, in_dim: int = 64, future_len: int = 1):
        super().__init__()
        self.future_len = int(future_len)
        self.proj = nn.Linear(int(in_dim), self.future_len * 2)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        y = self.proj(feat)
        return y.view(feat.size(0), self.future_len, 2)
