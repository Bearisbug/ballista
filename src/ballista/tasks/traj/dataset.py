from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_PIL_Image = None


def _get_pil_image():
    global _PIL_Image
    if _PIL_Image is not None:
        return _PIL_Image
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要安装 pillow 才能读取 png 图片：pip install pillow") from e
    _PIL_Image = Image
    return _PIL_Image


DatasetMode = Literal["reality", "synthesis", "both"]


@dataclass(frozen=True)
class FrameLabel:
    x_px: float
    y_px: float
    visibility: int  # 0/1


@dataclass
class ClipIndex:
    source: Literal["reality", "synthesis"]
    clip_dir: Path
    frames_dir: Path
    frame_files: List[str]
    labels_by_frame: Dict[str, FrameLabel]
    orig_w: int
    orig_h: int


def _is_clip_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit() and len(p.name) == 3


def _sorted_png_files(frames_dir: Path) -> List[str]:
    files = [x.name for x in frames_dir.iterdir() if x.is_file() and x.suffix.lower() == ".png"]
    files.sort()
    return files


def _read_image_size(path: Path) -> Tuple[int, int]:
    Image = _get_pil_image()
    with Image.open(path) as im:
        return im.size[0], im.size[1]  # (w, h)


def _load_and_resize_image(path: Path, out_w: int, out_h: int) -> torch.Tensor:
    Image = _get_pil_image()
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _norm_xy(x_px: float, y_px: float, w: int, h: int) -> Tuple[float, float]:
    den_w = max(1.0, float(w - 1))
    den_h = max(1.0, float(h - 1))
    return float(x_px) / den_w, float(y_px) / den_h


def _gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    if sigma <= 0:
        return heat
    if not (0.0 <= cx < w and 0.0 <= cy < h):
        return heat

    radius = int(3.0 * sigma)
    x0 = int(cx)
    y0 = int(cy)

    left = max(0, x0 - radius)
    right = min(w - 1, x0 + radius)
    top = max(0, y0 - radius)
    bottom = min(h - 1, y0 + radius)

    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    heat[top:bottom + 1, left:right + 1] = np.maximum(heat[top:bottom + 1, left:right + 1], g)
    return heat


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        kk = (k or "").strip().lstrip("\ufeff")
        vv = "" if v is None else str(v).strip()
        if kk:
            out[kk] = vv
    return out


def _frame_id_to_png_name(frame_id: str) -> str:
    s = (frame_id or "").strip()
    if not s:
        return ""
    name = Path(s).name
    # 允许 frame_id 只给数字：00000000
    stem = Path(name).stem
    if stem.isdigit():
        return f"{int(stem):08d}.png"
    if not name.lower().endswith(".png"):
        return name + ".png"
    return name


def _read_reality_annotations(
    csv_path: Path,
    frame_files: List[str],
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
) -> Dict[str, FrameLabel]:
    labels: Dict[str, FrameLabel] = {}
    frame_set = set(frame_files)

    scale_x = out_w / float(orig_w)
    scale_y = out_h / float(orig_h)

    # utf-8-sig 可以自动吞掉 BOM
    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _sanitize_row(raw)

            frame_str = row.get("Frame", row.get("frame", ""))
            if not frame_str.isdigit():
                continue
            frame_idx = int(frame_str)
            fname = f"{frame_idx:08d}.png"
            if fname not in frame_set:
                continue

            vis = int(float(row.get("Visibility", row.get("visibility", "1")) or 0))
            x = float(row.get("X", row.get("x", "0")) or 0)
            y = float(row.get("Y", row.get("y", "0")) or 0)

            x_r = _clamp(x * scale_x, 0.0, out_w - 1.0)
            y_r = _clamp(y * scale_y, 0.0, out_h - 1.0)
            labels[fname] = FrameLabel(x_px=x_r, y_px=y_r, visibility=vis)

    for fname in frame_files:
        if fname not in labels:
            labels[fname] = FrameLabel(x_px=0.0, y_px=0.0, visibility=0)
    return labels


def _read_synthesis_annotations(
    csv_path: Path,
    frame_files: List[str],
    orig_w: int,
    orig_h: int,
    out_w: int,
    out_h: int,
) -> Dict[str, FrameLabel]:
    labels: Dict[str, FrameLabel] = {}
    frame_set = set(frame_files)

    scale_x = out_w / float(orig_w)
    scale_y = out_h / float(orig_h)

    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _sanitize_row(raw)

            fname = _frame_id_to_png_name(row.get("frame_id", row.get("Frame", row.get("frame", ""))))
            if not fname or fname not in frame_set:
                continue

            # 只用 ball_u / ball_v
            if "ball_u" not in row or "ball_v" not in row:
                continue

            u = float(row.get("ball_u", "0") or 0)
            v_bottom = float(row.get("ball_v", "0") or 0)

            # 左下角原点 -> 左上角原点
            y_top = (orig_h - 1.0) - v_bottom

            x_r = _clamp(u * scale_x, 0.0, out_w - 1.0)
            y_r = _clamp(y_top * scale_y, 0.0, out_h - 1.0)
            labels[fname] = FrameLabel(x_px=x_r, y_px=y_r, visibility=1)

    for fname in frame_files:
        if fname not in labels:
            labels[fname] = FrameLabel(x_px=0.0, y_px=0.0, visibility=0)
    return labels


def _build_clip_indices(root: Path, source: Literal["reality", "synthesis"], out_w: int, out_h: int) -> List[ClipIndex]:
    if not root.exists():
        raise FileNotFoundError(f"{source} root not found: {root}")

    clips: List[ClipIndex] = []
    for clip_dir in sorted([p for p in root.iterdir() if _is_clip_dir(p)], key=lambda p: p.name):
        frames_dir = clip_dir / "frames"
        ann_path = clip_dir / "annotations.csv"
        if not frames_dir.exists() or not ann_path.exists():
            continue

        frame_files = _sorted_png_files(frames_dir)
        if len(frame_files) == 0:
            continue

        orig_w, orig_h = _read_image_size(frames_dir / frame_files[0])

        if source == "reality":
            labels = _read_reality_annotations(ann_path, frame_files, orig_w, orig_h, out_w, out_h)
        else:
            labels = _read_synthesis_annotations(ann_path, frame_files, orig_w, orig_h, out_w, out_h)

        clips.append(
            ClipIndex(
                source=source,
                clip_dir=clip_dir,
                frames_dir=frames_dir,
                frame_files=frame_files,
                labels_by_frame=labels,
                orig_w=orig_w,
                orig_h=orig_h,
            )
        )
    return clips


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        reality_root: Optional[str],
        synthesis_root: Optional[str],
        mode: DatasetMode = "reality",
        out_size: Tuple[int, int] = (512, 288),  # (W,H)
        seq_len: int = 8,
        sliding_step: int = 1,
        sigma: float = 2.5,
    ):
        super().__init__()
        self.mode = mode
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        self.seq_len = int(seq_len)
        self.sliding_step = int(sliding_step)
        self.sigma = float(sigma)

        self._clips: List[ClipIndex] = []

        if mode in ("reality", "both"):
            if not reality_root:
                raise ValueError("mode=reality/both 需要提供 reality_root")
            self._clips.extend(_build_clip_indices(Path(reality_root), "reality", self.out_w, self.out_h))

        if mode in ("synthesis", "both"):
            if not synthesis_root:
                raise ValueError("mode=synthesis/both 需要提供 synthesis_root")
            self._clips.extend(_build_clip_indices(Path(synthesis_root), "synthesis", self.out_w, self.out_h))

        if len(self._clips) == 0:
            raise RuntimeError(f"未找到任何 clip（mode={mode}），请检查数据路径与目录结构")

        self._index: List[Tuple[int, int]] = []
        for ci, clip in enumerate(self._clips):
            n = len(clip.frame_files)
            if n < self.seq_len:
                continue
            for start in range(0, n - self.seq_len + 1, self.sliding_step):
                self._index.append((ci, start))

        if len(self._index) == 0:
            raise RuntimeError("没有可用的滑窗样本（可能 seq_len 太大或目录为空）")

    def __len__(self) -> int:
        return len(self._index)

    def get_source_of_index(self, idx: int) -> str:
        ci, _ = self._index[idx]
        return self._clips[ci].source

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ci, start = self._index[idx]
        clip = self._clips[ci]
        frames = clip.frame_files[start : start + self.seq_len]

        images: List[torch.Tensor] = []
        heatmaps: List[torch.Tensor] = []
        coords_px: List[List[float]] = []
        coords_norm: List[List[float]] = []
        vis_list: List[int] = []

        for fname in frames:
            img_path = clip.frames_dir / fname
            img_t = _load_and_resize_image(img_path, self.out_w, self.out_h)
            images.append(img_t)

            lab = clip.labels_by_frame.get(fname, FrameLabel(0.0, 0.0, 0))
            x_px, y_px, vis = float(lab.x_px), float(lab.y_px), int(lab.visibility)

            xn, yn = _norm_xy(x_px, y_px, self.out_w, self.out_h)
            coords_px.append([x_px, y_px])
            coords_norm.append([xn, yn])
            vis_list.append(vis)

            if vis == 1:
                hm = _gaussian_heatmap(self.out_h, self.out_w, x_px, y_px, self.sigma)
            else:
                hm = np.zeros((self.out_h, self.out_w), dtype=np.float32)
            heatmaps.append(torch.from_numpy(hm).unsqueeze(0))

        return {
            "images": torch.stack(images, dim=0),
            "heatmaps": torch.stack(heatmaps, dim=0),
            "coords_px": torch.tensor(coords_px, dtype=torch.float32),
            "coords_norm": torch.tensor(coords_norm, dtype=torch.float32),
            "visibility": torch.tensor(vis_list, dtype=torch.int64),
            "source": clip.source,
            "clip_id": clip.clip_dir.name,
            "frame_ids": frames,
        }
