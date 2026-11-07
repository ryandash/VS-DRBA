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
    trt=False,
    gmflow_trt=False,
    rife_trt=False,
    profile_dir=None,
)
```

Set `trt=True` to enable TensorRT for FeatureNet, MetricNet, and FusionNet. On `feature/gmflow-tensorrt`, `gmflow_trt=True` additionally targets GMFlow's CNN backbone, while `rife_trt=True` targets the RIFE 4.25 encoder and IFBlocks. The switches are independent. Engines compile lazily on first use and specialize to fixed padded/pyramid dimensions, so startup can be substantially slower. Set `profile_dir=r".../traces"` to emit a PyTorch Chrome trace JSON for each processed output frame; use a short clip because profiling adds substantial overhead and disk usage. Torch-TensorRT partitions supported regions; GMFlow transformer attention, unsupported operators, DRM, SoftSplat, and VapourSynth orchestration remain eager. FP16 and FP32 are permitted, but this does not guarantee every layer runs in TensorRT. Keep all flags disabled for the eager baseline.

The input must be a constant-format RGBH or RGBS VapourSynth clip, contain at least two frames, and have CUDA available. The default checkpoint directory is `vsdrba/gmfss_union/weights/train_log_gmfss_union`; pass `weights=...` to override it. `factor_num/factor_den` controls the output frame-rate multiplier. Alternatively, specify both `fps_num` and `fps_den` to request an absolute output frame rate.

`sc=True` uses existing `_SceneChangeNext` frame properties. Supply `sc_threshold` to run VapourSynth's `misc.SCDetect` first. `linear=True` selects the linear DRM variant.

TensorRT is opt-in. On this branch, `trt=True` targets FeatureNet, MetricNet, and FusionNet; `gmflow_trt=True` targets GMFlow's CNN backbone; and `rife_trt=True` targets the RIFE 4.25 encoder and IFBlocks. GMFlow's transformer remains eager because Torch-TensorRT export currently fails on its partitioned-attention reshape/stride pattern. MetricNet and RIFE may still contain eager fallback regions. With all flags disabled, the original eager PyTorch/CUDA path is retained. The adapter pads frame dimensions to multiples of 32 and crops the result back to the source dimensions. GMFlow additionally pads its scaled half-resolution inputs to multiples of 32 before attention-window partitioning, then resizes/crops the flow to the original half-resolution dimensions. This makes reduced `scale` values such as `0.5` satisfy the model's [2, 8] attention split constraints. It serializes model inference through a lock because the first implementation prioritizes correctness over concurrent frame throughput.

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

This branch now has independent experimental TensorRT targets for FeatureNet, the convolution-only MetricNet head, FusionNet, GMFlow's CNN backbone, and the RIFE 4.25 encoder/IFBlocks. GMFlow's transformer, DRM, SoftSplat, masks, and VapourSynth frame orchestration remain eager. The initial hardware target is the GeForce RTX 5070 Ti (Blackwell), batch 1, FP16/FP32, fixed spatial shapes. These paths are experimental until component parity, engine-build reliability, steady-state throughput, and peak VRAM are measured on real clips.

## Validation status

- [x] Port GMFlow, FeatureNet, MetricNet, FusionNet, DRM, SoftSplat, and auxiliary RIFE integration.
- [x] Load the five GMFSS Union checkpoints.
- [x] Run a 256x256 synthetic tensor smoke test on an RTX 5070 Ti.
- [x] Add a separate VapourSynth clip adapter with rational frame-rate scheduling and optional scene-change handling.
- [ ] Validate clip output against real consecutive frames and inspect visual quality.
- [ ] Validate 1080p and 4K performance, VRAM, and frame scheduling.
- [ ] Compare TensorRT FeatureNet/FusionNet outputs against eager outputs on real 1080p and 4K clips.
- [ ] Measure steady-state throughput and peak VRAM with TensorRT enabled versus eager execution.
- [ ] Validate MetricNet TensorRT partitions against eager MetricNet.
- [ ] Validate bidirectional GMFlow flow ordering and parity against two eager directional passes.
- [ ] Validate reduced-scale GMFlow padding and output parity/quality on real clips.
- [ ] Benchmark the scaled-dot-product attention path against the original matmul/softmax implementation on the target GPU.
- [ ] Validate RIFE encoder/IFBlock TensorRT outputs against eager RIFE 4.25.
- [ ] Benchmark each TensorRT flag combination independently; compilation overhead and fallback partitions may erase expected gains.

### TensorRT expansion strategy

Treat each flag as an independent experiment. Start with one target at a time, warm up every fixed-shape engine, and inspect profiler traces to distinguish TensorRT engine execution from eager fallback. Compare component outputs against eager execution before combining flags. Bidirectional GMFlow should be compared against two directional calls, including forward/backward ordering; the center-frame feature cache should be compared against independently extracted features. Keep DRM and SoftSplat eager because they include warping semantics and custom tensor operations.


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
