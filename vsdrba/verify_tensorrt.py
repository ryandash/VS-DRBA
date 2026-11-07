"""Smoke-test the TensorRT-RTX backend in the active Python environment.

Run from the repository root:
    python -m vsdrba.verify_tensorrt

This builds and executes a small fixed-shape FP16 engine. It intentionally
rejects standard TensorRT/Torch-TensorRT installations.
"""

from __future__ import annotations

import importlib.metadata
import sys

import torch
import tensorrt_rtx
import torch_tensorrt  # noqa: F401 - registers the TensorRT-RTX backend


def main() -> int:
    distributions = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    if "torch-tensorrt" in distributions or "tensorrt" in distributions:
        raise RuntimeError(
            "Standard TensorRT packages are installed. Remove torch-tensorrt and "
            "the standard tensorrt distributions; this repository targets "
            "torch-tensorrt-rtx exclusively."
        )
    if "torch-tensorrt-rtx" not in distributions:
        raise RuntimeError("torch-tensorrt-rtx is not installed.")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Torch-TensorRT-RTX: {distributions['torch-tensorrt-rtx']}")
    print(f"TensorRT-RTX runtime: {tensorrt_rtx.__version__}")
    print(f"TensorRT-RTX module: {tensorrt_rtx.__file__}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this Python interpreter.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
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
            "pass_through_build_failures": True,
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
        raise RuntimeError("TensorRT-RTX output contains NaN or Inf values.")

    max_error = (actual - expected).abs().max().item()
    print(f"Output shape: {tuple(actual.shape)}")
    print(f"Maximum absolute error: {max_error:.6g}")
    print("PASS: TensorRT-RTX engine compiled and executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
