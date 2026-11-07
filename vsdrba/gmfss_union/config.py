"""Configuration primitives for GMFSS Union TensorRT engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class TensorRTConfig:
    """Per-network TensorRT build settings.

    GMFSS is compiled network-by-network. Keep warping, DRM, and VapourSynth
    frame orchestration in eager PyTorch/CUDA until the model port is validated.
    """

    precision: Literal["fp16", "fp32"] = "fp16"
    static_shape: bool = True
    optimization_level: int = 5
    workspace_size: int = 0
    max_aux_streams: int | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.optimization_level not in range(6):
            raise ValueError("optimization_level must be between 0 and 5")
        if self.workspace_size < 0:
            raise ValueError("workspace_size must be non-negative")
        if self.max_aux_streams is not None and self.max_aux_streams < 0:
            raise ValueError("max_aux_streams must be non-negative")
