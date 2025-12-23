from __future__ import annotations

import os
import sys
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

def _make_stdio_nonblocking(drop_on_block: bool = True) -> None:
    import os, sys
    for s in (sys.stdout, sys.stderr):
        try: os.set_blocking(s.fileno(), False)
        except Exception: pass

    class _W:
        def __init__(self, s): self.s = s
        def __getattr__(self, n): return getattr(self.s, n)
        def isatty(self):
            try: return self.s.isatty()
            except Exception: return False
        def write(self, d):
            try: return self.s.write(d)
            except BlockingIOError:
                if drop_on_block: return 0
                raise
            except Exception: return 0
        def flush(self):
            try: self.s.flush()
            except Exception: pass

    sys.stdout, sys.stderr = _W(sys.stdout), _W(sys.stderr)

_make_stdio_nonblocking(drop_on_block=True)

class Trainer:
    def __init__(self, cfg: Dict[str, Any], file_logger, console: Optional[Console], run_logger: RunLogger):
        self.cfg = cfg
        self.file_logger = file_logger
        self.run_logger = run_logger

        self.epochs = int(cfg["train"]["epochs"])
        self.log_interval = int(cfg["train"].get("log_interval", 50))

        out_dir = Path(cfg["output"]["dir"]) / cfg["output"]["exp_name"]
        self.ckpt_dir = out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if console is None:
            self.console = Console(file=sys.stdout, stderr=sys.stderr)
        else:
            self.console = Console(
                file=sys.stdout,
                stderr=sys.stderr,
                force_terminal=sys.stdout.isatty(),
                color_system=getattr(console, "color_system", "auto"),
                width=getattr(console, "width", None),
                soft_wrap=getattr(console, "soft_wrap", False),
            )

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
                        progress.console.log(f"epoch={epoch} step={global_step} loss={loss_str}")
                        self.file_logger.info(f"epoch={epoch} step={global_step} loss={loss_str}")
                        self.run_logger.log({"train/loss": loss_val, "train/epoch": epoch}, step=global_step)

                save_checkpoint(str(self.ckpt_dir / f"epoch_{epoch}.pt"), model, optimizer, epoch)
                progress.console.log(f"saved checkpoint: epoch_{epoch}.pt")
                self.file_logger.info(f"saved checkpoint: epoch_{epoch}.pt")
                # self.run_logger.log({"train/epoch_end": epoch}, step=global_step)
