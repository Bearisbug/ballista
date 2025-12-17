import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib import gridspec

from ballista.tasks.traj.dataset import TrajectoryDataset


def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.clip(x, 0.0, 1.0)
    return x


def _tensor_to_hm(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().numpy()
    if x.ndim == 3:
        x = x[0]
    x = np.clip(x, 0.0, 1.0)
    return x


def _draw_point(ax, x, y):
    ax.scatter([x], [y], s=30, color="yellow", alpha=0.5)


def _sample_indices_by_source(ds: TrajectoryDataset, source: str, k: int) -> list[int]:
    idxs = [i for i in range(len(ds)) if ds.get_source_of_index(i) == source]
    if len(idxs) < k:
        return idxs
    return random.sample(idxs, k)


def _save_single_pair(out_path: Path, img: np.ndarray, hm: np.ndarray, x_px: float, y_px: float, vis: int, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes

    ax0.imshow(img)
    if vis == 1:
        _draw_point(ax0, x_px, y_px)
    ax0.set_title(title)
    ax0.set_xlabel(f"X={x_px:.2f}, Y={y_px:.2f}, Visibility={vis}")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.imshow(hm, cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax1.set_title("heatmap")
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@pytest.mark.parametrize("mode", ["reality", "synthesis", "both"])
def test_traj_dataset_read_and_visualize(mode: str):
    random.seed(0)

    reality_root = os.environ.get("BALLISTA_REALITY_ROOT", "")
    synthesis_root = os.environ.get("BALLISTA_SYNTHESIS_ROOT", "")

    if mode in ("reality", "both") and not reality_root:
        pytest.skip("未设置 BALLISTA_REALITY_ROOT，跳过 reality/both 测试")
    if mode in ("synthesis", "both") and not synthesis_root:
        pytest.skip("未设置 BALLISTA_SYNTHESIS_ROOT，跳过 synthesis/both 测试")

    ds = TrajectoryDataset(
        reality_root=reality_root if reality_root else None,
        synthesis_root=synthesis_root if synthesis_root else None,
        mode=mode,
        out_size=(512, 288),
        seq_len=8,
        sliding_step=4,
        sigma=2.5,
    )
    assert len(ds) > 0

    picked: list[int] = []
    if mode == "both":
        picked.extend(_sample_indices_by_source(ds, "reality", 2))
        picked.extend(_sample_indices_by_source(ds, "synthesis", 2))
        if len(picked) < 4:
            all_idx = list(range(len(ds)))
            remain = [i for i in all_idx if i not in picked]
            picked.extend(remain[: (4 - len(picked))])
    else:
        picked = random.sample(list(range(len(ds))), k=min(4, len(ds)))
        if len(picked) < 4:
            picked = picked + picked[: (4 - len(picked))]

    out_dir = Path("tests") / "traj" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 总览：2行 × 2个样本；每个样本内部是 [img | hm]
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0],  # 每个样本占2列：img+hm
        wspace=0.02, hspace=0.05
    )

    for i, idx in enumerate(picked[:4]):
        r = i // 2          # 0 or 1
        c0 = (i % 2) * 2     # 0 or 2（每个样本占两列）

        sample = ds[idx]
        mid = sample["images"].shape[0] // 2

        img = _tensor_to_img(sample["images"][mid])
        hm = _tensor_to_hm(sample["heatmaps"][mid])
        x_px, y_px = sample["coords_px"][mid].tolist()
        vis = int(sample["visibility"][mid].item())

        title = f"source={sample['source']} clip={sample['clip_id']} frame={sample['frame_ids'][mid]}"

        ax_img = fig.add_subplot(gs[r, c0])
        ax_hm = fig.add_subplot(gs[r, c0 + 1])

        ax_img.imshow(img)
        if vis == 1:
            _draw_point(ax_img, x_px, y_px)
        ax_img.set_title(title, fontsize=10)
        ax_img.set_xlabel(f"X={x_px:.2f}, Y={y_px:.2f}, Vis={vis}", fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_hm.imshow(hm, cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_hm.set_title("heatmap", fontsize=10)
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])

        # ✅ 单样本也保存一份（image+heatmap）
        single_path = out_dir / f"dataset_pair_{mode}_{i}_{sample['source']}_{sample['clip_id']}_{sample['frame_ids'][mid]}.png"
        _save_single_pair(single_path, img, hm, x_px, y_px, vis, title)
        assert single_path.exists()

    out_path = out_dir / f"dataset_vis_{mode}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    assert out_path.exists()
