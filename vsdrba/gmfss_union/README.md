# GMFSS Union port

This package is the independent implementation boundary for GMFSS Union. It is deliberately separate from `drba_rife()`: GMFSS Union uses GMFlow + MetricNet for its motion/metric fields, and its DRM must not be confused with the RIFE block-0 DRM path.

## Inference graph

1. Three consecutive RGB frames are prepared and padded to model-compatible dimensions.
2. GMFlow estimates half-resolution motion from the center frame to its two neighbors.
3. FeatureNet extracts multiscale features; MetricNet estimates per-direction metric maps.
4. GMFSS DRM combines flow-distance maps and learned metrics to form the spatial timestep maps.
5. SoftSplat performs eager CUDA/PyTorch warping and hole handling.
6. RIFE 4.26 Heavy supplies the auxiliary intermediate prediction.
7. FusionNet combines warped RGB, the auxiliary prediction, and aligned multiscale features into the full-resolution output.

## TensorRT boundary

Compile independent engines for GMFlow, FeatureNet, MetricNet, RIFE 4.26 Heavy, and FusionNet. Keep DRM, SoftSplat, masks, frame caching, and VapourSynth scheduling outside TensorRT for the first correctness pass. Do not attempt a monolithic exported GMFSS graph.

Initial target: GeForce RTX 5070 Ti (Blackwell), batch 1, FP16, fixed shapes. Validate each engine against eager PyTorch before enabling it in the VapourSynth frame path. Engine cache identity must include model/checkpoint identity, GPU name, TensorRT and Torch-TensorRT versions, precision, and input dimensions.

## Milestone 1 status

- [x] Isolate the GMFSS Union package boundary.
- [x] Define shared TensorRT build settings for the per-network engines.
- [x] Retain only RIFE 4.25, 4.26, and 4.26 Heavy in the public RIFE selector and downloader.
- [x] Remove other RIFE implementation modules.
- [ ] Port the original GMFlow, FeatureNet, MetricNet, FusionNet, and GMFSS orchestration modules.
- [ ] Add and verify all five GMFSS Union checkpoints.
- [ ] Add eager PyTorch parity tests and per-network TensorRT export tests.
- [ ] Integrate the VapourSynth public API after eager inference is correct.
