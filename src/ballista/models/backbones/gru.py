import torch
import torch.nn as nn
from ballista.models.registry import BACKBONES


@BACKBONES.register("gru_backbone")
class GRUBackbone(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rnn = nn.GRU(
            input_size=int(input_dim),
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)   # [B, T, H]
        return out[:, -1, :]   # [B, H]
