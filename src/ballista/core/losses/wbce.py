import torch
import torch.nn as nn
from ballista.core.registry import LOSSES

@LOSSES.register("wbce_loss")
class WBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy loss (TrackNetV2 style).

    Args:
        reduction (str): "mean" | "none"
        eps (float): clamp epsilon for numerical stability
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got: {reduction}")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (N, C, H, W), values in [0, 1]
            y:      (N, C, H, W), values in {0, 1} (or [0,1] soft labels)

        Returns:
            if reduction=="mean": scalar tensor
            if reduction=="none": (N,) per-sample loss
        """
        # clamp to avoid log(0)
        p = torch.clamp(y_pred, self.eps, 1.0)
        q = torch.clamp(1.0 - y_pred, self.eps, 1.0)

        loss = -(
            torch.square(1.0 - y_pred) * y * torch.log(p)
            + torch.square(y_pred) * (1.0 - y) * torch.log(q)
        )

        if self.reduction == "mean":
            return loss.mean()
        else:
            # per-sample mean over (C,H,W)
            return loss.flatten(start_dim=1).mean(dim=1)
