import torch.nn as nn
from ballista.models.registry import NECKS


@NECKS.register("identity_neck")
class IdentityNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Identity()

    def forward(self, x):
        return self.op(x)
