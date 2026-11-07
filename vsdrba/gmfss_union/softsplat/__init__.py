"""Select the CUDA SoftSplat kernel when CuPy is available, else PyTorch."""

try:
    import cupy  # noqa: F401
    from .softsplat import softsplat
except Exception:
    from .softsplat_torch import softsplat

__all__ = ["softsplat"]
