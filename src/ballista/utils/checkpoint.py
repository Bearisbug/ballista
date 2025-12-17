from pathlib import Path
import torch


def save_checkpoint(path: str, model, optimizer, epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
