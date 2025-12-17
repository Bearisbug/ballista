from typing import Any, Dict

import torch.nn as nn

from ballista.core.registry import LOSSES, MODELS


def _build_from_cfg(cfg: Dict[str, Any], registry) -> Any:
    name = cfg.get("name")
    if not name:
        raise ValueError("cfg must have field: name")
    cls = registry.get(name)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, MODELS)


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, LOSSES)
