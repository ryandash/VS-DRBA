# VS-DRBA
Distance Ratio Based Adjuster for Animeinterp, based on https://github.com/routineLife1/DRBA and https://github.com/HolyWu/vs-rife.

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DRBA](https://github.com/routineLife1/DRBA) project.
>
> With TensorRT integration, it achieves a 400% acceleration, enabling real-time playback on high-performance NVIDIA GPUs.
>

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)
- [cupy](https://github.com/cupy/cupy) cupy-cuda11x or later (critical for acceleration)

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.7.0.post1 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) matching the installed PyTorch version and CUDA build

To install the latest stable version of PyTorch, Torch-TensorRT and cupy, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com
pip install -U cupy-cuda12x
```

## Installation
```
pip install -U vsdrba==1.0.2
```
The supported RIFE checkpoints are 4.25, 4.26, and 4.26 Heavy. To download these checkpoints, run `python -m vsdrba`. The GMFSS Union port is being developed in the `gmfss_union` package on the `feature/gmfss-union-milestone-1` branch.

## Usage
```python
from vsdrba import drba_rife

ret = drba_rife(clip)
```

See `__init__.py` for the description of the parameters.


## Benchmarks

| model                | scale | os    | hardware           | arch                                                       | fps 720 | fps 1080 | vram 720 | vram 1080 | backend                                                                        | verified output                    | batch | level | streams | threads | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                               |
|----------------------| ----- | ----- |--------------------|------------------------------------------------------------|---------|----------|----------|-----------|--------------------------------------------------------------------------------|------------------------------------|-------|-------|---------|---------|-----------|--------------------|-------------|---------------|-----------|-----------------------------------------------------------------------------------------------------|
| rife 4.26 heavy      | 2x    | Linux | 3070laptop / 12400 | [rife](https://github.com/hzwer/Practical-RIFE) (4.26)     | 119     | 53       | 1.6gb    | 3.4gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (holywu vsrife)       | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |
| drba_rife 4.26 heavy | 2x    | Linux | 3070laptop / 12400 | [drba_rife](https://github.com/routineLife1/DRBA) (4.26)   | 158     | 70       | 1.7gb    | 3.7gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (routineLife1 vsdrba) | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |


## Model support and GMFSS Union

This branch intentionally retains only the RIFE 4.25, 4.26, and 4.26 Heavy implementations. GMFSS Union uses the RIFE 4.25 architecture as its auxiliary interpolation network, independently of the public RIFE model selector.

GMFSS Union is being ported as a separate pipeline rather than as another RIFE selector. Its first integration stages are GMFlow, FeatureNet, MetricNet, DRM, SoftSplat, the RIFE 4.25 auxiliary network, and FusionNet. TensorRT compilation will be introduced per network, with PyTorch/CUDA warping and DRM kept outside TensorRT until correctness is established.

Target validation hardware: NVIDIA GeForce RTX 5070 Ti (Blackwell). Initial TensorRT validation should use fixed spatial dimensions and FP16, with engine cache keys including GPU, TensorRT/PyTorch versions, resolution, and precision.


## GMFSS Union VapourSynth API

GMFSS Union is available as a separate pipeline and does not add a new value to the RIFE `model` selector:

```python
from vsdrba import drba_gmfss

clip = drba_gmfss(
    clip,
    factor_num=2,
    factor_den=1,
    scale=1.0,
    sc=True,
    trt=False,
    gmflow_trt=False,
)
```

The GMFSS clip path currently requires RGBH or RGBS input and CUDA. `trt=True` experimentally compiles FeatureNet and FusionNet through Torch-TensorRT. On the experimental `feature/gmflow-tensorrt` branch, `gmflow_trt=True` additionally targets GMFlow's convolutional backbone only; its transformer, local matching, and grid sampling remain eager. The first interpolated frame triggers lazy engine compilation for the padded clip dimensions. Keep `trt=False` for the baseline path. See [the GMFSS Union guide](vsdrba/gmfss_union/README.md) for checkpoint paths, frame-rate options, scene-change handling, and validation limits.


## Standard TensorRT on Windows (CUDA 12.8)

Torch-TensorRT must match the installed PyTorch build. For example, the following package combination targets a PyTorch `2.11.0+cu128` environment:

- `torch==2.11.0+cu128`
- `torch-tensorrt==2.11.0+cu128`
- Standard NVIDIA TensorRT `10.15.1.29` CUDA 12 runtime packages (`tensorrt-cu12`, `tensorrt-cu12-bindings`, and `tensorrt-cu12-libs`)

Do not install TensorRT-RTX for this configuration. Also avoid leaving the CUDA 13 TensorRT metapackage/runtime installed alongside the CUDA 12 runtime: the generic `tensorrt` metapackage may select the CUDA 13 distribution even though this PyTorch/Torch-TensorRT build targets CUDA 12.8.

In the portable Python environment, remove the generic/CUDA 13 distributions while retaining the CUDA 12 TensorRT packages required by Torch-TensorRT:

```bat
python -m pip uninstall -y tensorrt tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs
```

Then verify the CUDA 12 runtime is still importable:

```bat
python -c "import torch, tensorrt, torch_tensorrt; print('PyTorch:', torch.__version__); print('TensorRT:', tensorrt.__version__); print('TensorRT module:', tensorrt.__file__)"
python -m pip check
```

Run the repository smoke test to verify that Torch-TensorRT can actually compile and execute a small FP16 convolution through the `torch.compile` backend (an import-only check does not prove that engine compilation works):

```bat
python scripts\verify_tensorrt.py
```

If the smoke test fails, keep `trt=False` / `gmflow_trt=False` and provide the complete traceback. The eager PyTorch path does not require TensorRT. These commands are specific to the stated CUDA 12.8 environment; do not apply them to a CUDA 13 PyTorch installation.
