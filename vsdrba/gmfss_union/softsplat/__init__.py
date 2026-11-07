"""Select CuPy SoftSplat when its CUDA runtime is available, else PyTorch."""

try:
    import cupy

    if cupy.cuda.get_cuda_path() is None:
        raise RuntimeError("CuPy cannot locate a CUDA toolkit")
    from .softsplat import softsplat
except Exception:
    from .softsplat_torch import softsplat

__all__ = ["softsplat"]
