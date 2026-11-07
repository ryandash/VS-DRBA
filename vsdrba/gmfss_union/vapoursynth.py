"""VapourSynth clip adapter for the GMFSS Union tensor pipeline."""

from __future__ import annotations

import math
import os
from fractions import Fraction
from threading import Lock
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .model import GMFSS_UNION


def _frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    planes = [
        torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
        for plane in range(frame.format.num_planes)
    ]
    return torch.stack(planes).unsqueeze(0)


def _tensor_to_frame(
    tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream
) -> vs.VideoFrame:
    tensor = tensor.squeeze(0).detach()
    planes = [tensor[i].to("cpu", non_blocking=True) for i in range(frame.format.num_planes)]
    stream.synchronize()
    output = frame.copy()
    for i, plane in enumerate(planes):
        np.copyto(np.asarray(output[i]), plane.numpy())
    return output


def drba_gmfss(
    clip: vs.VideoNode,
    device_index: int = 0,
    weights: str | os.PathLike | None = None,
    rife_weights: str | os.PathLike | None = None,
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    sc: bool = False,
    sc_threshold: float | None = None,
    linear: bool = False,
    trt: bool = False,
) -> vs.VideoNode:
    """Interpolate an RGB VapourSynth clip with GMFSS Union.

    With trt=True, FeatureNet and FusionNet are compiled lazily by
    Torch-TensorRT for the clip's fixed padded dimensions. GMFlow, MetricNet,
    auxiliary RIFE 4.25, DRM, and SoftSplat remain on the eager PyTorch path.

    The auxiliary RIFE model loads the standard flownet_v4.25.pkl checkpoint
    from vsdrba/models by default. Use rife_weights to override its location.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("gmfss: input is not a VapourSynth clip")
    if clip.format is None or clip.format.id not in (vs.RGBH, vs.RGBS):
        raise vs.Error("gmfss: only constant-format RGBH and RGBS clips are supported")
    if clip.num_frames < 2:
        raise vs.Error("gmfss: clip must contain at least two frames")
    if not torch.cuda.is_available():
        raise vs.Error("gmfss: CUDA is not available")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise vs.Error("gmfss: device_index is outside the available CUDA devices")
    if factor_num < 1 or factor_den < 1:
        raise vs.Error("gmfss: factor_num and factor_den must be positive")
    if scale not in (0.25, 0.5, 1.0, 2.0, 4.0):
        raise vs.Error("gmfss: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")
    if sc_threshold is not None and not 0.0 <= sc_threshold <= 1.0:
        raise vs.Error("gmfss: sc_threshold must be between 0.0 and 1.0")
    if (fps_num is None) != (fps_den is None):
        raise vs.Error("gmfss: fps_num and fps_den must be specified together")
    if fps_num is not None:
        if fps_num < 1 or fps_den < 1:
            raise vs.Error("gmfss: fps_num and fps_den must be positive")
        if clip.fps == 0:
            raise vs.Error("gmfss: clip must have a valid frame rate when fps_num/fps_den are used")
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()
    if factor_num <= factor_den:
        raise vs.Error("gmfss: target frame rate must be higher than source frame rate")
    package_weights = Path(__file__).resolve().parent / "weights" / "train_log_gmfss_union"
    if not package_weights.is_dir():
        # Source checkout layout keeps the checkpoints at repository-root weights/.
        package_weights = Path(__file__).resolve().parents[2] / "weights" / "train_log_gmfss_union"
    weights_path = Path(weights).expanduser() if weights is not None else package_weights
    if not weights_path.is_dir():
        raise vs.Error(f"gmfss: checkpoint directory does not exist: {weights_path}")
    required = ("flownet.pkl", "metric.pkl", "feat.pkl", "fusionnet.pkl")
    missing = [name for name in required if not (weights_path / name).is_file()]
    if missing:
        raise vs.Error(f"gmfss: missing checkpoint files in {weights_path}: {', '.join(missing)}")

    package_rife_weights = Path(__file__).resolve().parents[1] / "models" / "flownet_v4.25.pkl"
    rife_weights_path = (
        Path(rife_weights).expanduser() if rife_weights is not None else package_rife_weights
    )
    if not rife_weights_path.is_file():
        raise vs.Error(
            "gmfss: RIFE 4.25 checkpoint not found at "
            f"{rife_weights_path}. Run python -m vsdrba to download it, "
            "or pass rife_weights=... explicitly."
        )

    if sc and sc_threshold is not None:
        scene_clip = clip.resize.Bicubic(format=vs.GRAY8, matrix_s="709").misc.SCDetect(sc_threshold)

        source_clip = clip

        def copy_scene_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
            output = f[0].copy()
            output.props["_SceneChangePrev"] = f[1].props.get("_SceneChangePrev", 0)
            output.props["_SceneChangeNext"] = f[1].props.get("_SceneChangeNext", 0)
            return output

        clip = source_clip.std.ModifyFrame(
            [source_clip, scene_clip], copy_scene_props
        )

    device = torch.device("cuda", device_index)
    torch.set_float32_matmul_precision("high")
    try:
        model = GMFSS_UNION(
            weights=str(weights_path), scale=scale, device=device, trt=trt,
            rife_weights=str(rife_weights_path),
        )
    except Exception as exc:
        if trt:
            raise vs.Error(f"gmfss: TensorRT initialization failed: {exc}") from exc
        raise
    width, height = clip.width, clip.height
    padded_width = math.ceil(width / 32) * 32
    padded_height = math.ceil(height / 32) * 32
    padding = (0, padded_width - width, 0, padded_height - height)
    stream = torch.cuda.Stream(device=device)
    inference_lock = Lock()

    @torch.inference_mode()
    def interpolate(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        t = (n * factor_den % factor_num) / factor_num
        if t == 0 or (sc and f[0].props.get("_SceneChangeNext", 0)):
            return f[0]

        with inference_lock, torch.cuda.stream(stream):
            image0, image1, image2 = [
                _frame_to_tensor(frame, device) for frame in f
            ]
            if any(padding):
                image0 = F.pad(image0, padding)
                image1 = F.pad(image1, padding)
                image2 = F.pad(image2, padding)

            outputs, _ = model.inference_ts_drba(
                image0, image1, image2, [t], linear=linear
            )
            output = outputs[0]
            if any(padding):
                output = output[:, :, :height, :width]
            return _tensor_to_frame(output, f[0], stream)

    clip0 = vs.core.std.Interleave([clip] * factor_num)
    clip1 = clip.std.DuplicateFrames(clip.num_frames - 1)[1:]
    clip1 = vs.core.std.Interleave([clip1] * factor_num)
    clip2 = (
        clip.std.DuplicateFrames(clip.num_frames - 1)
        .std.DuplicateFrames(clip.num_frames - 1)[2:]
    )
    clip2 = vs.core.std.Interleave([clip2] * factor_num)
    if factor_den > 1:
        clip0, clip1, clip2 = clip0[::factor_den], clip1[::factor_den], clip2[::factor_den]

    return clip0.std.ModifyFrame([clip0, clip1, clip2], interpolate)


__all__ = ["drba_gmfss"]
