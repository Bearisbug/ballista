from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Column

from ballista.utils.checkpoint import save_checkpoint
from ballista.utils.logging import RunLogger


class Trainer:
    def __init__(self, cfg: Dict[str, Any], file_logger, console: Console, run_logger: RunLogger):
        self.cfg = cfg
        self.file_logger = file_logger
        self.console = console
        self.run_logger = run_logger

        self.epochs = int(cfg["train"]["epochs"])
        self.log_interval = int(cfg["train"].get("log_interval", 50))

        out_dir = Path(cfg["output"]["dir"]) / cfg["output"]["exp_name"]
        self.ckpt_dir = out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, task, model, train_loader, optimizer):
        global_step = 0
        try:
            total_batches: Optional[int] = len(train_loader)
        except TypeError:
            total_batches = None

        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", table_column=Column(width=10, no_wrap=True)),
            TextColumn("epoch {task.fields[epoch]:>2}/{task.fields[epochs]:>2}", table_column=Column(width=18, no_wrap=True)),
            BarColumn(bar_width=40),
            MofNCompleteColumn(table_column=Column(width=12, justify="right")),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("loss {task.fields[loss]:>10}", table_column=Column(width=16, justify="right", no_wrap=True)),
            console=self.console,
            transient=False,
            refresh_per_second=10,
            redirect_stdout=True,
            redirect_stderr=True,
        )

        with progress:
            task_id = progress.add_task("Train", total=total_batches, epoch=1, epochs=self.epochs, loss="-")

            for epoch in range(1, self.epochs + 1):
                model.train()

                progress.reset(
                    task_id,
                    total=total_batches,
                    completed=0,
                    description="Train",
                    start=True,
                    epoch=epoch,
                    epochs=self.epochs,
                    loss="-",
                )

                for batch in train_loader:
                    loss, logs = task.train_step(batch, model)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    loss_val = float(logs.get("loss", 0.0))
                    loss_str = f"{loss_val:.6f}"

                    progress.update(task_id, loss=loss_str)
                    progress.advance(task_id, 1)

                    if self.log_interval > 0 and (global_step % self.log_interval == 0):
                        # 控制台：只在上方输出，不影响进度条（rich live 的特性）
                        progress.console.log(f"epoch={epoch} step={global_step} loss={loss_str}")
                        # 文件日志
                        self.file_logger.info(f"epoch={epoch} step={global_step} loss={loss_str}")
                        # 外部 tracker（wandb/swanlab）
                        self.run_logger.log({"train/loss": loss_val, "train/epoch": epoch}, step=global_step)

                save_checkpoint(str(self.ckpt_dir / f"epoch_{epoch}.pt"), model, optimizer, epoch)
                progress.console.log(f"saved checkpoint: epoch_{epoch}.pt")
                self.file_logger.info(f"saved checkpoint: epoch_{epoch}.pt")
                # self.run_logger.log({"train/epoch_end": epoch}, step=global_step)
