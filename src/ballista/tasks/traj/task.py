from typing import Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ballista.core  
from ballista.core.build import build_loss, build_model
from ballista.tasks.base import BaseTask
from ballista.tasks.traj.transforms import build_traj_transforms


class TrajTask(BaseTask):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        # 可选：走 registry loss；不想就直接 self.loss_fn = nn.MSELoss()
        if "loss" in cfg and cfg["loss"] is not None:
            self.loss_fn = build_loss(cfg["loss"])
        else:
            self.loss_fn = nn.MSELoss()

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        model = build_model(cfg["model"])
        return model.to(self.device)

    # def build_dataloader(self, cfg: Dict[str, Any], split: str):
    #     dcfg = cfg["dataset"]
    #     ds = SyntheticTrajectoryDataset(
    #         num_samples=int(dcfg["num_samples"]),
    #         past_len=int(dcfg["past_len"]),
    #         future_len=int(dcfg["future_len"]),
    #         transform=build_traj_transforms(noise_std=float(dcfg.get("noise_std", 0.0))),
    #     )
    #     return DataLoader(
    #         ds,
    #         batch_size=int(dcfg["batch_size"]),
    #         shuffle=(split == "train"),
    #         num_workers=int(dcfg.get("num_workers", 0)),
    #         drop_last=(split == "train"),
    #     )

    def build_optimizer(self, cfg: Dict[str, Any], model: nn.Module):
        ocfg = cfg["optimizer"]
        lr = float(ocfg["lr"])
        wd = float(ocfg.get("weight_decay", 0.0))
        opt_type = ocfg.get("type", "adamw").lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if opt_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt_type}")

    def train_step(self, batch: Dict[str, Any], model: nn.Module):
        past = batch["past"].to(self.device).float()
        future = batch["future"].to(self.device).float()

        pred = model(past)
        loss = self.loss_fn(pred, future)

        return loss, {"loss": float(loss.detach().cpu())}
