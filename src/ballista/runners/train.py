from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from ballista.core.seed import set_seed
from ballista.core.logging import setup_experiment_logging
from ballista.engine.trainer import Trainer
from ballista.tasks.linreg.task import LinearRegressionTask
from ballista.tasks.traj.task import TrajTask


TASKS = {
    "traj": TrajTask,
    "linreg": LinearRegressionTask,
}


def _pick_device(device_cfg: Dict[str, Any]) -> torch.device:
    dtype = str(device_cfg.get("type", "cpu")).lower()
    if dtype == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dtype == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def build_task(cfg: Dict[str, Any], device: torch.device):
    task_name = str(cfg.get("task_name", "") or "")
    cls = TASKS.get(task_name)
    if cls is None:
        raise ValueError(f"Unknown task_name={task_name}. Available={sorted(TASKS.keys())}")
    return cls(cfg, device=device)


def run(cfg: DictConfig) -> None:
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )  # type: ignore[assignment]
    set_seed(int(cfg_dict.get("seed", 42)))

    file_logger, console, run_logger = setup_experiment_logging(cfg_dict)

    out_dir = Path(cfg_dict["output"]["dir"]) / cfg_dict["output"]["exp_name"]
    console.log(f"Composed config via Hydra. Output dir: {out_dir}")

    device = _pick_device(cfg_dict.get("device", {}) or {})
    console.log(f"Device: {device}")

    task = build_task(cfg_dict, device=device)
    model = task.build_model(cfg_dict)
    train_loader = task.build_dataloader(cfg_dict, split="train")
    optimizer = task.build_optimizer(cfg_dict, model)

    Trainer(cfg_dict, file_logger=file_logger, console=console, run_logger=run_logger).fit(
        task, model, train_loader, optimizer
    )

    run_logger.close()
