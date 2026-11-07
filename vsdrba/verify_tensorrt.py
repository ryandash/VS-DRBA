"""Smoke-test the standard Torch-TensorRT backend on the active Python environment.

Run from the repository root:
    python scripts/verify_tensorrt.py

This deliberately builds a tiny fixed-shape FP16 engine. It is a backend
validation test, not a performance benchmark.
"""

from __future__ import annotations

import importlib.metadata
import sys

import torch
import tensorrt
import torch_tensorrt  # noqa: F401 - registers the torch.compile backend


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torch-TensorRT distribution: {importlib.metadata.version('torch-tensorrt')}")
    print(f"TensorRT runtime: {tensorrt.__version__}")
    print(f"TensorRT module: {tensorrt.__file__}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this Python interpreter.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    # Exercise the native TensorRT builder before invoking Torch-TensorRT.
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    _builder = tensorrt.Builder(logger)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    ).eval().cuda().half()
    sample = torch.randn(1, 3, 128, 128, device="cuda", dtype=torch.float16)

    compiled = torch.compile(
        model,
        backend="tensorrt",
        dynamic=False,
        options={
            "enabled_precisions": {torch.float16},
            "optimization_level": 3,
            "min_block_size": 1,
        },
    )

    with torch.inference_mode():
        expected = model(sample)
        actual = compiled(sample)
        torch.cuda.synchronize()

    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Output shape mismatch: expected {tuple(expected.shape)}, "
            f"got {tuple(actual.shape)}"
        )
    if not torch.isfinite(actual).all().item():
        raise RuntimeError("TensorRT output contains NaN or Inf values.")

    max_error = (actual - expected).abs().max().item()
    print(f"Output shape: {tuple(actual.shape)}")
    print(f"Maximum absolute error: {max_error:.6g}")
    print("PASS: standard TensorRT engine compiled and executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
