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
)
```

Set `trt=True` to enable the experimental TensorRT path for FeatureNet and FusionNet. Their engines compile lazily on the first interpolated frame and are specialized to the padded clip dimensions. The first frame can therefore take substantially longer while engines are built. GMFlow, MetricNet, auxiliary RIFE, DRM, and SoftSplat remain on eager PyTorch/CUDA. The initial implementation allows FP16 and FP32 TensorRT precisions; it does not promise that every operation in either network becomes a TensorRT engine. Keep `trt=False` as the baseline and fallback.

The input must be a constant-format RGBH or RGBS VapourSynth clip, contain at least two frames, and have CUDA available. The default checkpoint directory is `vsdrba/gmfss_union/weights/train_log_gmfss_union`; pass `weights=...` to override it. `factor_num/factor_den` controls the output frame-rate multiplier. Alternatively, specify both `fps_num` and `fps_den` to request an absolute output frame rate.

`sc=True` uses existing `_SceneChangeNext` frame properties. Supply `sc_threshold` to run VapourSynth's `misc.SCDetect` first. `linear=True` selects the linear DRM variant.

TensorRT is opt-in and currently targets FeatureNet and FusionNet only. `trt=False` retains the original eager PyTorch/CUDA execution path. The adapter pads frame dimensions to multiples of 32 and crops the result back to the source dimensions. It serializes model inference through a lock because the first implementation prioritizes correctness over concurrent frame throughput.

## Inference graph

1. Three consecutive RGB frames are prepared and padded.
2. GMFlow estimates half-resolution motion from the center frame to its neighbors.
3. FeatureNet extracts multiscale features; MetricNet estimates per-direction metric maps.
4. GMFSS DRM forms spatial timestep maps.
5. SoftSplat performs CUDA/PyTorch warping and hole handling.
6. RIFE 4.26 Heavy supplies the auxiliary intermediate prediction.
7. FusionNet combines warped RGB, the auxiliary prediction, and aligned multiscale features.

The RIFE auxiliary checkpoint includes `teacher.*` and `caltime.*` training-only parameters. Those keys are intentionally ignored; all inference parameters must load, and any other unexpected key causes initialization to fail.

## TensorRT boundary

The intended future export targets are GMFlow, FeatureNet, MetricNet, RIFE 4.26 Heavy, and FusionNet independently. DRM, SoftSplat, masks, and VapourSynth frame orchestration remain outside TensorRT until eager inference and numerical parity are established. The initial hardware target is the GeForce RTX 5070 Ti (Blackwell), batch 1, FP16, fixed spatial shapes.

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
