from typing import Any, Dict
import torch.nn as nn

from ballista.models.registry import BACKBONES, NECKS, HEADS, LOSSES, ARCHS


def _build_from_cfg(cfg: Dict[str, Any], registry) -> Any:
    name = cfg.get("name")
    if not name:
        raise ValueError("cfg must have field: name")
    cls = registry.get(name)
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)


def build_backbone(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, BACKBONES)


def build_neck(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, NECKS)


def build_head(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, HEADS)


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    return _build_from_cfg(cfg, LOSSES)


def build_arch(cfg: Dict[str, Any]) -> nn.Module:
    """
    model cfg:
      name: multi_head
      backbone: {...}
      neck: {...} (optional)
      heads: [...]
      head_keys: [...]
    """
    name = cfg.get("name")
    if not name:
        raise ValueError("model cfg must have field: name")

    cls = ARCHS.get(name)

    backbone = build_backbone(cfg["backbone"])
    neck = build_neck(cfg["neck"]) if "neck" in cfg and cfg["neck"] is not None else None
    heads = [build_head(h) for h in cfg["heads"]]

    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "backbone", "neck", "heads", "output_key")}
    return cls(backbone=backbone, neck=neck, heads=heads, **kwargs)
