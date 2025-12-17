from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class LinearRegressionDataset(Dataset):
    """
    简单的线性回归玩具数据：y = w^T x + b + noise。
    默认在初始化时固定一组 w/b，方便训练可复现。
    """

    def __init__(
        self,
        num_samples: int = 512,
        input_dim: int = 1,
        weight: Optional[Sequence[float]] = None,
        bias: Optional[float] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.num_samples = int(num_samples)
        self.input_dim = int(input_dim)
        self.noise_std = float(noise_std)

        if weight is None:
            self.weight = np.random.uniform(-2.0, 2.0, size=(self.input_dim,))
        else:
            w_arr = np.asarray(weight, dtype=np.float32)
            if w_arr.shape != (self.input_dim,):
                raise ValueError(f"weight shape must be ({self.input_dim},), got {w_arr.shape}")
            self.weight = w_arr

        self.bias = float(np.random.uniform(-1.0, 1.0) if bias is None else bias)

        self._x = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
        noise = (np.random.randn(self.num_samples) * self.noise_std).astype(np.float32)
        self._y = (self._x @ self.weight.reshape(-1, 1)).squeeze(1) + self.bias + noise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self._x[idx]
        y = self._y[idx]
        return {"x": torch.from_numpy(x), "y": torch.tensor(y, dtype=torch.float32)}
