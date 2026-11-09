# VS-DRBA WIP with some bugs
Distance Ratio Based Adjuster for Animeinterp, based on https://github.com/routineLife1/DRBA and https://github.com/HolyWu/vs-rife.

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DRBA](https://github.com/routineLife1/DRBA) project.
>
> With TensorRT-RTX integration, it targets accelerated inference, enabling real-time playback on high-performance NVIDIA GPUs.
>

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.11.x (CUDA 12.9 build for the documented RTX setup)
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)
- [cupy](https://github.com/cupy/cupy) cupy-cuda11x or later (critical for acceleration)

`trt=True` requires [TensorRT-RTX](https://developer.nvidia.com/tensorrt-rtx) and [Torch-TensorRT-RTX](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html). Standard TensorRT is not supported by this branch. The documented Windows configuration uses Python 3.13, PyTorch 2.11, and CUDA 12.9.

## Installation
```
python -m pip install hatchling
python -m pip install -U "vsdrba @ git+https://github.com/ryandash/VS-DRBA.git"
python -m vsdrba
```
The supported RIFE checkpoints are 4.25, 4.26, and 4.26 Heavy. To download these checkpoints, run `python -m vsdrba`. GMFSS Union is implemented in the `vsdrba/gmfss_union` package on this branch.

## Usage
```python
from vsdrba import drba_rife

clip = drba_rife(
    clip,
    factor_num=2,
    factor_den=1,
    model="4.25",
    sc=True,
    sc_threshold=0.15,
    trt=True,
    trt_optimization_level=5
)
```
or
```python
from vsgmfss_fortuna import gmfss_fortuna

clip = gmfss_fortuna(
    clip,
    factor_num=2,
    factor_den=1,
    model=1,
    scale=1.0,
    sc=True,
    sc_threshold=0.15,
    num_streams=1,
    trt=True,
    trt_optimization_level=5
)
```

See `__init__.py` for the description of the parameters.

## TensorRT-RTX on Windows

This branch targets **Torch-TensorRT-RTX 2.11.0**, **PyTorch 2.11.0**, **CUDA 12.9**, and **TensorRT-RTX 1.3**. Install the CUDA 12.9 Update 1 Toolkit and a driver compatible with it. The CUDA Toolkit is a system prerequisite; the CUDA runtime bundled with PyTorch is not a substitute for the required Toolkit installation. Verify with `nvcc --version`.

Use a separate portable Python environment or back up the existing one before changing runtimes. Remove standard TensorRT packages first; do not mix the standard and RTX Python runtimes:

```bat
python -m pip uninstall -y torch-tensorrt tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs
```

Install the matching PyTorch CUDA 12.9 wheels and Torch-TensorRT-RTX:

```bat
python -m pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.11.0+cu129 torchvision==0.26.0+cu129
python -m pip install --index-url https://download.pytorch.org/whl/torch-tensorrt-rtx --extra-index-url https://pypi.org/simple torch-tensorrt-rtx==2.11.0
python -m pip install -U cupy-cuda12x
python -m pip check
```

If the RTX runtime wheel is not resolved for your platform, follow the Windows-specific installation instructions in the official [Torch-TensorRT-RTX guide](https://docs.pytorch.org/TensorRT/getting_started/tensorrt_rtx.html) and the NVIDIA [TensorRT-RTX prerequisites](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/prerequisites.html). Do not install the standard `tensorrt` or `torch-tensorrt` packages as a workaround.

Verify the runtime and compile a small FP16 engine:

```bat
python -c "import torch, torch_tensorrt, tensorrt_rtx; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Torch-TensorRT-RTX:', torch_tensorrt.__version__); print('TensorRT-RTX:', tensorrt_rtx.__version__); print('GPU:', torch.cuda.get_device_name(0))"
python -m vsdrba.verify_tensorrt
```

The smoke test rejects standard TensorRT distributions and fails if TensorRT-RTX cannot compile or execute its engine. For GMFSS Union, enable `trt=True`; compilation failures are surfaced rather than silently disabling acceleration. The deliberately uncompiled warping/DRM/orchestration operations remain PyTorch operations by design.
