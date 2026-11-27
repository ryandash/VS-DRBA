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

TensorRT inference uses the standard [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) and [Torch-TensorRT](https://pytorch.org/TensorRT/) packages. The CUDA extra installs the Python-side CUDA dependencies, including Torch-TensorRT and the CUDA 13.x CuPy package.

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


## Benchmarks

| model                | scale | os    | hardware           | arch                                                       | fps 720 | fps 1080 | vram 720 | vram 1080 | backend                                                                        | verified output                    | batch | level | streams | threads | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                               |
|----------------------| ----- | ----- |--------------------|------------------------------------------------------------|---------|----------|----------|-----------|--------------------------------------------------------------------------------| ---------------------------------- | ----- | ----- |---------|---------| --------- | ------------------ | ----------- | ------------- | --------- |-----------------------------------------------------------------------------------------------------|
| rife 4.26 heavy      | 2x    | Linux | 3070laptop / 12400 | [rife](https://github.com/hzwer/Practical-RIFE) (4.26)     | 119     | 53       | 1.6gb    | 3.4gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (holywu vsrife)       | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |
| drba_rife 4.26 heavy | 2x    | Linux | 3070laptop / 12400 | [drba_rife](https://github.com/routineLife1/DRBA) (4.26)   | 158     | 70       | 1.7gb    | 3.7gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (routineLife1 vsdrba) | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |
