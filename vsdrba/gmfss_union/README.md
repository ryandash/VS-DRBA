# GMFSS Union

GMFSS Union is exposed as a separate VapourSynth pipeline, not as a RIFE model selector. The existing `drba_rife()` API and its RIFE 4.25, 4.26, and 4.26 Heavy choices remain unchanged.

## VapourSynth usage

```python
from vsdrba.gmfss_union import drba_gmfss

clip = drba_gmfss(
    clip,
    factor_num=2,
    factor_den=1,
    scale=1.0,
    sc=True,
    trt=True,
    profile_dir=None,
)
```

Set `trt=True` to enable TensorRT-RTX for FeatureNet, the convolution-only MetricNet head, FusionNet, GMFlow's CNN backbone, and the RIFE 4.25 encoder. These are the only deliberately compiled submodules. GMFlow transformer/matching, DRM, SoftSplat, RIFE IFBlocks/warping, and VapourSynth orchestration remain outside the engine boundary by design. RTX import and engine-build failures are fatal; there is no standard TensorRT or automatic eager fallback. Each compiled submodule is saved as a `torch.export.raw TensorRT engine` TensorRT artifact (`.engine`) and loaded on later runs. A cache miss still requires a one-time build, and each artifact specializes to its padded/pyramid dimensions. The default cache is `~/.cache/vsdrba/tensorrt_rtx`; pass `trt_cache_dir=...` to override it. Cache keys include checkpoint metadata, shape, dtype, GPU model/compute capability, PyTorch version, and TensorRT-RTX version. The existing `.runtime_cache` files remain separate and are used during engine construction. Set `profile_dir=r".../traces"` to emit a PyTorch Chrome trace JSON for each processed output frame; use a short clip because profiling adds substantial overhead and disk usage.

The input must be a constant-format RGBH or RGBS VapourSynth clip, contain at least two frames, and have CUDA available. The default checkpoint directory is `vsdrba/gmfss_union/weights/train_log_gmfss_union`; pass `weights=...` to override it. `factor_num/factor_den` controls the output frame-rate multiplier. Alternatively, specify both `fps_num` and `fps_den` to request an absolute output frame rate.

`sc=True` uses existing `_SceneChangeNext` frame properties. Supply `sc_threshold` to run VapourSynth's `misc.SCDetect` first. `linear=True` selects the linear DRM variant.

TensorRT-RTX is opt-in through the single `trt=True` switch. Serialized artifacts use the Torch-TensorRT raw TensorRT engine save/load path and Python runtime; they are not portable across arbitrary GPU architectures or runtime versions. FeatureNet, MetricNet's convolution head, FusionNet, GMFlow's CNN backbone, and the RIFE encoder are exported and compiled as complete TensorRT-RTX engines with full-compilation enforcement. GMFlow's transformer/matching, RIFE IFBlocks/warping, DRM, SoftSplat, and VapourSynth orchestration remain outside the engine boundary by design. RTX import and engine-build errors raise immediately. Setting `trt=False` explicitly selects the uncompiled PyTorch path; it is not an automatic fallback. The adapter pads frame dimensions to multiples of 32 and crops the result back to the source dimensions. GMFlow additionally pads its scaled half-resolution inputs to multiples of 32 before attention-window partitioning, then resizes/crops the flow to the original half-resolution dimensions. This makes reduced `scale` values such as `0.5` satisfy the model's [2, 8] attention split constraints. GMFlow's transformer now uses PyTorch scaled-dot-product attention where available and caches shifted-window masks for repeated fixed-shape inference. These are performance experiments; compare output quality and steady-state FPS against the previous implementation on the target GPU. It serializes model inference through a lock because the first implementation prioritizes correctness over concurrent frame throughput.

## Inference graph

1. Three consecutive RGB frames are prepared and padded.
2. GMFlow estimates forward/backward half-resolution motion in one bidirectional pass for each center-to-neighbor pair.
3. FeatureNet extracts multiscale features once per frame; the center-frame features are reused across both neighboring pairs. MetricNet estimates per-direction metric maps.
4. GMFSS DRM forms spatial timestep maps.
5. SoftSplat performs CUDA/PyTorch warping and hole handling.
6. RIFE 4.25 supplies the auxiliary intermediate prediction.
7. FusionNet combines warped RGB, the auxiliary prediction, and aligned multiscale features.

The auxiliary network uses the standard RIFE 4.25 checkpoint `flownet_v4.25.pkl`, downloaded by `python -m vsdrba` into `vsdrba/models/`. Pass `rife_weights=...` to `drba_gmfss()` to select another compatible 4.25 checkpoint. The old GMFSS `rife.pkl` heavy checkpoint is no longer used.

## TensorRT boundary

This branch now has strict TensorRT-RTX targets for FeatureNet, the convolution-only MetricNet head, FusionNet, GMFlow's CNN backbone, and the RIFE 4.25 encoder. GMFlow's transformer, DRM, SoftSplat, masks, and VapourSynth frame orchestration remain eager. The initial hardware target is the GeForce RTX 5070 Ti (Blackwell), batch 1, FP16/FP32, fixed spatial shapes. These paths are experimental until component parity, engine-build reliability, steady-state throughput, and peak VRAM are measured on real clips.

## Validation status

- [x] Port GMFlow, FeatureNet, MetricNet, FusionNet, DRM, SoftSplat, and auxiliary RIFE integration.
- [x] Load the five GMFSS Union checkpoints.
- [x] Run a 256x256 synthetic tensor smoke test on an RTX 5070 Ti.
- [x] Add a separate VapourSynth clip adapter with rational frame-rate scheduling and optional scene-change handling.
- [ ] Validate clip output against real consecutive frames and inspect visual quality.
- [ ] Validate 1080p and 4K performance, VRAM, and frame scheduling.
- [ ] Compare TensorRT-RTX FeatureNet/FusionNet outputs against eager outputs on real 1080p and 4K clips.
- [ ] Measure steady-state throughput and peak VRAM with TensorRT-RTX enabled versus the explicit eager baseline.
- [ ] Validate MetricNet TensorRT-RTX engines against eager MetricNet.
- [ ] Validate bidirectional GMFlow flow ordering and parity against two eager directional passes.
- [ ] Validate reduced-scale GMFlow padding and output parity/quality on real clips.
- [ ] Benchmark the scaled-dot-product attention path and cached window masks against the original matmul/softmax implementation on the target GPU.
- [ ] Validate RIFE encoder TensorRT-RTX outputs against eager RIFE 4.25.
- [ ] Benchmark the complete TensorRT-RTX configuration; compilation overhead may erase expected gains.

### TensorRT-RTX expansion strategy

Warm up every fixed-shape engine and inspect profiler traces to confirm the expected complete module engines execute. Compare component outputs against the explicit eager baseline before measuring steady-state throughput. Bidirectional GMFlow should be compared against two directional calls, including forward/backward ordering; the center-frame feature cache should be compared against independently extracted features. Keep DRM and SoftSplat eager because they include warping semantics and custom tensor operations.


## RIFE auxiliary model selection

GMFSS Union now uses the RIFE 4.25 architecture for its auxiliary intermediate prediction network. It no longer imports RIFE 4.26 Heavy or loads the old `rife.pkl` file from the GMFSS checkpoint directory.

Download the standard RIFE checkpoints (including `flownet_v4.25.pkl`) with:

```bash
python -m vsdrba
```

The GMFSS Union adapter looks for the 4.25 checkpoint at `vsdrba/models/flownet_v4.25.pkl`. You can supply a compatible checkpoint at another location:

```python
clip = drba_gmfss(
    clip,
    weights=r".../train_log_gmfss_union",
    rife_weights=r".../flownet_v4.25.pkl",
    factor_num=2,
    factor_den=1,
)
```

The GMFSS checkpoint directory still needs `flownet.pkl`, `metric.pkl`, `feat.pkl`, and `fusionnet.pkl`; `rife.pkl` is no longer required. The 4.25 model has a smaller feature encoding (4 channels) than 4.26 Heavy (16 channels), so the two checkpoint formats are not interchangeable. Use weights trained for the 4.25 architecture; do not point `rife_weights` at the old heavy `rife.pkl`.
