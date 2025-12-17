from typing import Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ballista.core 

from ballista.core.build import build_loss, build_model
from ballista.tasks.base import BaseTask
from ballista.tasks.traj.dataset import TrajectoryDataset


def _collate_tracknet(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    metas: list[Dict[str, Any]] = []

    for sample in batch:
        images: torch.Tensor = sample["images"]  # [T, 3, H, W]
        heatmaps: torch.Tensor = sample["heatmaps"]  # [T, 1, H, W]

        if images.ndim != 4:
            raise ValueError(f"images must be 4D [T,3,H,W], got shape={tuple(images.shape)}")
        if heatmaps.ndim != 4:
            raise ValueError(f"heatmaps must be 4D [T,1,H,W], got shape={tuple(heatmaps.shape)}")

        t, c, h, w = images.shape
        x = images.reshape(t * c, h, w)  # [T*3, H, W]
        y = heatmaps.squeeze(1)  # [T, H, W]

        xs.append(x)
        ys.append(y)
        metas.append(
            {
                "source": sample.get("source"),
                "clip_id": sample.get("clip_id"),
                "frame_ids": sample.get("frame_ids"),
            }
        )

    return {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
        "meta": metas,
    }


class TrajTask(BaseTask):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        if "loss" in cfg and cfg["loss"] is not None:
            self.loss_fn = build_loss(cfg["loss"])
        else:
            self.loss_fn = nn.MSELoss()

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        model = build_model(cfg["model"])
        return model.to(self.device)

    def build_dataloader(self, cfg: Dict[str, Any], split: str):
        dcfg = cfg["dataset"]
        out_size_raw = dcfg.get("out_size", (512, 288))
        out_w, out_h = int(out_size_raw[0]), int(out_size_raw[1])

        reality_root = dcfg.get("reality_root")
        synthesis_root = dcfg.get("synthesis_root")
        mode = str(dcfg.get("mode", "reality")).lower()
        if mode == "both":
            if reality_root and not synthesis_root:
                mode = "reality"
            elif synthesis_root and not reality_root:
                mode = "synthesis"

        ds = TrajectoryDataset(
            reality_root=reality_root,
            synthesis_root=synthesis_root,
            mode=mode,
            out_size=(out_w, out_h),
            seq_len=int(dcfg.get("seq_len", 3)),
            sliding_step=int(dcfg.get("sliding_step", 1)),
            sigma=float(dcfg.get("sigma", 2.5)),
        )

        return DataLoader(
            ds,
            batch_size=int(dcfg.get("batch_size", 4)),
            shuffle=(split == "train"),
            num_workers=int(dcfg.get("num_workers", 0)),
            drop_last=(split == "train"),
            collate_fn=_collate_tracknet,
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
        raise ValueError(f"Unknown optimizer: {opt_type}")

    def train_step(self, batch: Dict[str, Any], model: nn.Module):
        x: torch.Tensor = batch["x"].to(self.device).float()  # [B, T*3, H, W]
        y: torch.Tensor = batch["y"].to(self.device).float()  # [B, T, H, W]

        pred: torch.Tensor = model(x)  # [B, T, H, W] (when out_dim == seq_len)
        if pred.shape != y.shape:
            raise ValueError(f"pred/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(y.shape)}")

        loss = self.loss_fn(pred, y)
        return loss, {"loss": float(loss.detach().cpu())}
