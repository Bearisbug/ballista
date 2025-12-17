from typing import Any, Dict, Callable, List
import numpy as np
import torch


class Compose:
    def __init__(self, transforms: List[Callable[[Dict[str, Any]], Dict[str, Any]]]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class AddGaussianNoise:
    def __init__(self, key: str, std: float):
        self.key = key
        self.std = float(std)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.std <= 0:
            return sample
        x = sample[self.key]
        noise = (np.random.randn(*x.shape).astype(np.float32)) * self.std
        sample[self.key] = x + noise
        return sample


class ToTensor:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
            else:
                out[k] = v
        return out


def build_traj_transforms(noise_std: float) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    return Compose([AddGaussianNoise(key="past", std=noise_std), ToTensor()])
