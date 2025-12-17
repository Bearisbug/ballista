from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ballista.tasks.base import BaseTask
from ballista.tasks.linreg.dataset import LinearRegressionDataset


class LinearRegressionTask(BaseTask):
    """最小的线性回归示例任务，演示 Hydra+Ballista 的跑通流程。"""

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.loss_fn = nn.MSELoss()

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        mcfg = cfg.get("model", {}) or {}
        input_dim = int(mcfg.get("input_dim", 1))
        output_dim = int(mcfg.get("output_dim", 1))
        if output_dim != 1:
            raise ValueError("LinearRegressionTask 目前仅支持 output_dim=1")
        model = nn.Linear(input_dim, output_dim)
        return model.to(self.device)

    def build_dataloader(self, cfg: Dict[str, Any], split: str):
        dcfg = cfg["dataset"]
        ds = LinearRegressionDataset(
            num_samples=int(dcfg["num_samples"]),
            input_dim=int(dcfg.get("input_dim", 1)),
            weight=dcfg.get("weight"),
            bias=dcfg.get("bias"),
            noise_std=float(dcfg.get("noise_std", 0.0)),
        )
        return DataLoader(
            ds,
            batch_size=int(dcfg.get("batch_size", 32)),
            shuffle=(split == "train"),
            num_workers=int(dcfg.get("num_workers", 0)),
            drop_last=(split == "train"),
        )

    def build_optimizer(self, cfg: Dict[str, Any], model: nn.Module):
        ocfg = cfg["optimizer"]
        lr = float(ocfg["lr"])
        wd = float(ocfg.get("weight_decay", 0.0))
        opt_type = ocfg.get("type", "adamw").lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if opt_type == "sgd":
            momentum = float(ocfg.get("momentum", 0.0))
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        raise ValueError(f"Unknown optimizer: {opt_type}")

    def train_step(self, batch: Dict[str, Any], model: nn.Module):
        x = batch["x"].to(self.device).float()
        y = batch["y"].to(self.device).float()

        pred = model(x).squeeze(-1)
        target = y.view_as(pred)

        loss = self.loss_fn(pred, target)
        mae = F.l1_loss(pred, target)

        return loss, {"loss": float(loss.detach().cpu()), "mae": float(mae.detach().cpu())}
