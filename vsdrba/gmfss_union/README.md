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
)
```

Set `trt=True` to enable the experimental TensorRT path for FeatureNet and FusionNet. On the experimental `feature/gmflow-tensorrt` branch, also set `gmflow_trt=True` to compile GMFlow's CNN backbone, transformer, flow-propagation attention, and convex upsampler independently. Their engines compile lazily on the first interpolated frame and are specialized to the padded clip dimensions. The first frame can therefore take substantially longer while engines are built. GMFlow matching, grid sampling, MetricNet, auxiliary RIFE, DRM, and SoftSplat remain on eager PyTorch/CUDA. The initial implementation allows FP16 and FP32 TensorRT precisions; it does not promise that every operation in either network becomes a TensorRT engine. Keep `trt=False` as the baseline and fallback.

The input must be a constant-format RGBH or RGBS VapourSynth clip, contain at least two frames, and have CUDA available. The default checkpoint directory is `vsdrba/gmfss_union/weights/train_log_gmfss_union`; pass `weights=...` to override it. `factor_num/factor_den` controls the output frame-rate multiplier. Alternatively, specify both `fps_num` and `fps_den` to request an absolute output frame rate.

`sc=True` uses existing `_SceneChangeNext` frame properties. Supply `sc_threshold` to run VapourSynth's `misc.SCDetect` first. `linear=True` selects the linear DRM variant.

TensorRT is opt-in. On this branch, `gmflow_trt=True` additionally targets selected GMFlow submodules. Compilation can fall back for unsupported subgraphs, so inspect runtime logs and benchmark rather than assuming the entire GMFlow graph is accelerated. `trt=False` retains the original eager PyTorch/CUDA execution path. The adapter pads frame dimensions to multiples of 32 and crops the result back to the source dimensions. It serializes model inference through a lock because the first implementation prioritizes correctness over concurrent frame throughput.

## Inference graph

1. Three consecutive RGB frames are prepared and padded.
2. GMFlow estimates half-resolution motion from the center frame to its neighbors.
3. FeatureNet extracts multiscale features; MetricNet estimates per-direction metric maps.
4. GMFSS DRM forms spatial timestep maps.
5. SoftSplat performs CUDA/PyTorch warping and hole handling.
6. RIFE 4.25 supplies the auxiliary intermediate prediction.
7. FusionNet combines warped RGB, the auxiliary prediction, and aligned multiscale features.

The auxiliary network uses the standard RIFE 4.25 checkpoint `flownet_v4.25.pkl`, downloaded by `python -m vsdrba` into `vsdrba/models/`. Pass `rife_weights=...` to `drba_gmfss()` to select another compatible 4.25 checkpoint. The old GMFSS `rife.pkl` heavy checkpoint is no longer used.

## TensorRT boundary

The intended future export targets are GMFlow, FeatureNet, MetricNet, RIFE 4.25, and FusionNet independently. DRM, SoftSplat, masks, and VapourSynth frame orchestration remain outside TensorRT until eager inference and numerical parity are established. The initial hardware target is the GeForce RTX 5070 Ti (Blackwell), batch 1, FP16, fixed spatial shapes.

## Validation status

- [x] Port GMFlow, FeatureNet, MetricNet, FusionNet, DRM, SoftSplat, and auxiliary RIFE integration.
- [x] Load the five GMFSS Union checkpoints.
- [x] Run a 256x256 synthetic tensor smoke test on an RTX 5070 Ti.
- [x] Add a separate VapourSynth clip adapter with rational frame-rate scheduling and optional scene-change handling.
- [ ] Validate clip output against real consecutive frames and inspect visual quality.
- [ ] Validate 1080p and 4K performance, VRAM, and frame scheduling.
- [ ] Compare TensorRT FeatureNet/FusionNet outputs against eager outputs on real 1080p and 4K clips.
- [ ] Measure steady-state throughput and peak VRAM with TensorRT enabled versus eager execution.
- [ ] Expand TensorRT coverage to MetricNet, GMFlow, and auxiliary RIFE only after component-level parity tests.

### TensorRT expansion strategy

Do not convert every remaining module at once. Use the traces to rank GMFlow, MetricNet, auxiliary RIFE, DRM, and SoftSplat by CUDA time. FeatureNet and FusionNet are passed through Torch-TensorRT, but that does not imply every operator is compiled: inspect the trace for fallback PyTorch operators and TensorRT engine execution. If GMFlow or MetricNet dominates, evaluate it as an isolated export target next; validate output parity on representative frames before enabling it in the full pipeline. DRM and SoftSplat include warping semantics and custom tensor operations, so keep them as explicit separate targets until numerical parity and performance are demonstrated.


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
