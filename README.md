# VS-DRBA
Distance Ratio Based Adjuster for Animeinterp, based on https://github.com/routineLife1/DRBA and https://github.com/HolyWu/vs-rife.

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DRBA](https://github.com/routineLife1/DRBA) project.
> 
> With TensorRT integration, it achieves a 400% acceleration, enabling real-time playback on high-performance NVIDIA GPUs.
>

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (required only when scene change detection is enabled)
- [CuPy](https://github.com/cupy/cupy) (installed by the CUDA extra)

`trt=True` requires these additional packages:

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) **11.4.x** (standard TensorRT; not TensorRT-RTX)
- [Torch-TensorRT](https://pytorch.org/TensorRT/) **2.15.0** or later, matched to your installed PyTorch and CUDA versions

The current Torch-TensorRT 2.15.0 Windows wheels target PyTorch 2.15, CUDA 13.2/13.4, and TensorRT 11.4. Install a mutually compatible PyTorch, CUDA, TensorRT, and Torch-TensorRT combination; these packages are version-coupled, so the newest version of each package is not necessarily a compatible set.

## Installation

Install the latest code and its CUDA dependencies from this repository's `main` branch:

```bash
python -m pip install "vsdrba[cuda] @ git+https://github.com/ryandash/VS-DRBA.git@main"
```

For a CPU-only installation, omit the `[cuda]` extra:

```bash
python -m pip install "vsdrba @ git+https://github.com/ryandash/VS-DRBA.git@main"
```

The CUDA extra does not select a specific PyTorch CUDA wheel index. If you need a particular CUDA build, install the matching PyTorch and Torch-TensorRT builds for your CUDA environment separately, then install VS-DRBA without forcing a reinstall of those packages.

If you want to download all models at once, run `python -m vsdrba`. To download only the model specified at first use, set `auto_download=True` in `drba_rife()`.

## Usage
```python
from vsdrba import drba_rife

ret = drba_rife(clip)
```

See `__init__.py` for the description of the parameters.
