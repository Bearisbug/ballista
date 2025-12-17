from __future__ import annotations

import torch
import torch.nn as nn

from ballista.core.registry import MODELS


@MODELS.register("gru_traj")
class GRUTrajModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        future_len: int = 1,
    ):
        super().__init__()
        self.future_len = int(future_len)

        hidden_dim = int(hidden_dim)
        self.rnn = nn.GRU(
            input_size=int(input_dim),
            hidden_size=hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, self.future_len * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.rnn(x)  # [B, T, H]
        last = feat[:, -1, :]  # [B, H]

        y = self.proj(last)
        y = y.view(x.size(0), self.future_len, 2)
        return y
