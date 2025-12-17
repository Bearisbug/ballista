import torch.nn as nn

from ballista.core.registry import LOSSES


@LOSSES.register("mse_loss")
class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.fn(pred, target)
