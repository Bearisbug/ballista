from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn


class BaseTask(ABC):
    @abstractmethod
    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        ...

    @abstractmethod
    def build_dataloader(self, cfg: Dict[str, Any], split: str):
        ...

    @abstractmethod
    def build_optimizer(self, cfg: Dict[str, Any], model: nn.Module):
        ...

    @abstractmethod
    def train_step(self, batch: Dict[str, Any], model: nn.Module):
        ...

    def val_step(self, batch: Dict[str, Any], model: nn.Module):
        return {}
