"""Small utilities shared by the GMFSS Union inference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F


def resize(tensor: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


def distance_calculator(flow: torch.Tensor) -> torch.Tensor:
    dtype = flow.dtype
    u, v = flow[:, 0:1].float(), flow[:, 1:2].float()
    return torch.sqrt(u.square() + v.square()).to(dtype)


_ones_cache: dict[tuple[str, str], torch.Tensor] = {}


def get_ones_tensor_size(size: tuple, device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(size))
    if key not in _ones_cache:
        _ones_cache[key] = torch.ones(size, device=device, dtype=dtype, requires_grad=False)
    return _ones_cache[key]


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a plain or wrapped checkpoint and normalize DataParallel prefixes."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint at {path} is not a state-dict mapping")

    state = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue
        normalized = key
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        state[normalized] = value
    if not state:
        raise ValueError(f"Checkpoint at {path} contains no parameter entries")
    return state
