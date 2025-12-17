from typing import Dict, List, Optional
import torch
import torch.nn as nn

from ballista.models.registry import ARCHS


@ARCHS.register("multi_head")
class MultiHeadArch(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        heads: List[nn.Module],
        neck: Optional[nn.Module] = None,
        head_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = nn.ModuleList(heads)
        self.head_keys = head_keys or [f"head{i}" for i in range(len(heads))]
        if len(self.head_keys) != len(self.heads):
            raise ValueError("head_keys length must match heads length")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if self.neck is not None:
            feat = self.neck(feat)
        out: Dict[str, torch.Tensor] = {}
        for k, head in zip(self.head_keys, self.heads):
            out[k] = head(feat)
        return out
