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

## Component-level profiling

Profiling is opt-in and runs inside the VapourSynth frame callback. The profiler is started and stopped on the same callback thread; do not wrap the entire VapourSynth render in an external `torch.profiler.profile` context. Profiling adds significant overhead, so do not use profiled runs to report normal playback FPS.

```python
from vsdrba import drba_gmfss

clip = drba_gmfss(
    clip,
    factor_num=2,
    factor_den=1,
    trt=True,
    profile_dir=r"D:\gmfss_traces",
    profile_frames=3,
    profile_skip_frames=4,
)

clip.set_output()
```

Render the clip using your usual VapourSynth host. The first interpolated frame triggers lazy TensorRT compilation; the default skip value is intended to avoid profiling that first build. Each selected interpolated frame writes a separate Chrome trace named `gmfss_frame_XXXXXX.json` into `profile_dir`. Set `profile_skip_frames=0` only when intentionally profiling cold-start and engine-build costs. `profile_frames` controls the number of traces captured.

Open the JSON files in Perfetto (https://ui.perfetto.dev/) or `chrome://tracing`. Search for the `GMFSS/` ranges. The component ranges include:

- `GMFSS/FeatureNet/frame0` and `frame1`: the two FeatureNet calls.
- `GMFSS/GMFlow/center_to_prev` and `center_to_next`: both optical-flow directions.
- `GMFSS/MetricNet`: MetricNet forward.
- `GMFSS/DRM/GMFSS` and `GMFSS/DRM/Auxiliary_RIFE`: the two DRM paths, including tensor warps.
- `GMFSS/Auxiliary_RIFE/Feature_Encode` and `IFNet_Forward`: auxiliary RIFE.
- `GMFSS/SoftSplat/feature_and_rgb_warps`: RGB and multiscale feature warps.
- `GMFSS/FusionNet`: final fusion network.

The trace also includes CPU and CUDA operator events, input shapes, and memory activity. Compare CUDA kernel time inside each component range; do not sum parent and child durations. The captured frame is a steady-state sample only if TensorRT engine compilation has completed. Run an unprofiled render separately for true FPS.

### TensorRT expansion strategy

Do not convert every remaining module at once. Use the traces to rank GMFlow, MetricNet, auxiliary RIFE, DRM, and SoftSplat by CUDA time. FeatureNet and FusionNet are passed through Torch-TensorRT, but that does not imply every operator is compiled: inspect the trace for fallback PyTorch operators and TensorRT engine execution. If GMFlow or MetricNet dominates, evaluate it as an isolated export target next; validate output parity on representative frames before enabling it in the full pipeline. DRM and SoftSplat include warping semantics and custom tensor operations, so keep them as explicit separate targets until numerical parity and performance are demonstrated.
