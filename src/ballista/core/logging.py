from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console


class RunLogger:
    """统一的实验记录接口（W&B / SwanLab / none）"""
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullRunLogger(RunLogger):
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        return

    def close(self) -> None:
        return


class WandbRunLogger(RunLogger):
    def __init__(self, init_kwargs: Dict[str, Any]):
        import wandb  # lazy import
        self._wandb = wandb
        # wandb.init(project=..., name=..., dir=..., config=...)
        self.run = wandb.init(**init_kwargs)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        # Run.log 默认每次调用会推进 step；也可显式传 step
        if step is None:
            self.run.log(metrics)
        else:
            self.run.log(metrics, step=step)

    def close(self) -> None:
        try:
            self.run.finish()
        except Exception:
            pass


class SwanLabRunLogger(RunLogger):
    def __init__(self, init_kwargs: Dict[str, Any]):
        import swanlab  # lazy import
        self._swanlab = swanlab
        # swanlab.init(...) 后再 swanlab.log(...)
        self.run = swanlab.init(**init_kwargs)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        # swanlab.log 支持 step 参数
        if step is None:
            self._swanlab.log(metrics)
        else:
            self._swanlab.log(metrics, step=step)

    def close(self) -> None:
        try:
            # 文档/示例里常用 swanlab.finish() 结束实验
            self._swanlab.finish()
        except Exception:
            pass


def _build_run_logger(cfg: Dict[str, Any], out_dir: Path, console: Console) -> RunLogger:
    tcfg = ((cfg.get("logging") or {}).get("tracker")) or {}
    enable = bool(tcfg.get("enable", False))
    backend = str(tcfg.get("backend", "none")).lower()
    init_kwargs = dict(tcfg.get("init") or {})

    if not enable or backend in ("none", "null", ""):
        return NullRunLogger()

    # 给常用字段补默认值：name/dir/logdir
    exp_name = str(cfg["output"]["exp_name"])
    if backend == "wandb":
        init_kwargs.setdefault("name", exp_name)
        init_kwargs.setdefault("dir", str(out_dir / "wandb"))
        try:
            return WandbRunLogger(init_kwargs)
        except Exception as e:
            console.log(f"[yellow]W&B 初始化失败（可能没安装 wandb 或未登录）：{e}[/yellow]")
            return NullRunLogger()

    if backend == "swanlab":
        init_kwargs.setdefault("experiment_name", exp_name)
        init_kwargs.setdefault("logdir", str(out_dir / "swanlog"))
        try:
            return SwanLabRunLogger(init_kwargs)
        except Exception as e:
            console.log(f"[yellow]SwanLab 初始化失败（可能没安装 swanlab 或未登录）：{e}[/yellow]")
            return NullRunLogger()

    console.log(f"[yellow]未知 tracker backend={backend}，已禁用[/yellow]")
    return NullRunLogger()


def setup_experiment_logging(cfg: Dict[str, Any]) -> Tuple[logging.Logger, Console, RunLogger]:
    out_dir = Path(cfg["output"]["dir"]) / cfg["output"]["exp_name"]
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    console = Console(stderr=True)

    # 文件日志：稳定不影响 Rich Progress
    file_logger = logging.getLogger("ballista")
    file_logger.setLevel(logging.INFO)
    file_logger.handlers.clear()
    file_logger.propagate = False

    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setLevel(file_logger.level)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    file_logger.addHandler(fh)

    run_logger = _build_run_logger(cfg, out_dir=out_dir, console=console)
    return file_logger, console, run_logger
