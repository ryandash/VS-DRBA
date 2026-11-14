"""GMFSS Union tensor inference and RIFE auxiliary adapter."""

import copy
import logging
import os
import re
import warnings

import torch
import torch.nn.functional as F

from .GMFSS import Model
from .drm import calc_drm_gmfss, calc_drm_rife_auxiliary
from .utils import load_state_dict, resize
from ..IFNet_HDv3_v4_25 import IFNet

warnings.filterwarnings("ignore", ".*were assigned during export.*")


def _import_tensorrt_rtx():
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    logging.getLogger("torch._opaque_base").setLevel(logging.ERROR)
    logging.getLogger("torch._library.opaque_object").setLevel(logging.ERROR)

    import tensorrt_rtx
    import torch_tensorrt
    from torch_tensorrt.dynamo.conversion.impl.grid import GridSamplerInterpolationMode

    GridSamplerInterpolationMode.update(
        {
            0: tensorrt_rtx.InterpolationMode.LINEAR,
            1: tensorrt_rtx.InterpolationMode.NEAREST,
            2: tensorrt_rtx.InterpolationMode.CUBIC,
        }
    )

    torch_tensorrt.logging.set_level(logging.ERROR)
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    return tensorrt_rtx, torch_tensorrt


class _StrictTensorRTRTXModule(torch.nn.Module):
    """Lazily export and compile one fixed-shape module with full RTX coverage."""

    def __init__(
        self,
        module,
        label,
        cache_dir=None,
        cache_key="default",
        use_explicit_typing=False,
        free_module_after_compile=True,
    ):
        super().__init__()
        self.module = module
        self.label = label
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        self.use_explicit_typing = use_explicit_typing
        self.free_module_after_compile = free_module_after_compile
        self.compiled = None

    def forward(self, *inputs):
        if self.compiled is None:
            try:
                tensorrt_rtx, torch_tensorrt = _import_tensorrt_rtx()

                target_device = inputs[0].device
                self.module.to(target_device)

                exported = torch.export.export(self.module, tuple(inputs))
                shape_key = "_".join(
                    "x".join(str(int(dim)) for dim in tensor.shape)
                    for tensor in inputs
                )
                dtype_key = str(inputs[0].dtype).replace("torch.", "")
                device_key = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_",
                    torch.cuda.get_device_name(target_device),
                )
                cache_path = None
                if self.cache_dir is not None:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    cache_name = (
                        self.label + "_" + self.cache_key + "_" + shape_key + "_"
                        + dtype_key + "_" + device_key + "_trt-"
                        + str(tensorrt_rtx.__version__) + ".runtime_cache"
                    )
                    cache_path = os.path.join(self.cache_dir, cache_name)

                compile_kwargs = dict(
                    inputs=list(inputs),
                    optimization_level=5,
                    require_full_compilation=True,
                    pass_through_build_failures=True,
                    runtime_cache_path=cache_path,
                )
                if self.use_explicit_typing:
                    compile_kwargs["use_explicit_typing"] = True
                else:
                    compile_kwargs["enabled_precisions"] = {torch.float16}

                if self.cache_dir is not None:
                    compile_kwargs.update(
                        cache_built_engines=True,
                        reuse_cached_engines=True,
                        engine_cache_dir=self.cache_dir,
                    )

                self.compiled = torch_tensorrt.dynamo.compile(
                    exported,
                    **compile_kwargs,
                ).eval()

                if self.free_module_after_compile:
                    self.module.to("cpu")
                    torch.cuda.empty_cache()
            except Exception as exc:
                raise RuntimeError(
                    f"TensorRT-RTX could not fully compile {self.label}; "
                    f"no PyTorch fallback is permitted: {exc}"
                ) from exc
        return self.compiled(*inputs)


class _GMFlowBidirWrapper(torch.nn.Module):
    """Fixed-argument wrapper so full bidirectional GMFlow compiles into TensorRT-RTX."""

    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet

    def forward(self, img0, img1):
        return self.flownet(img0, img1, pred_bidir_flow=True)


class GMFSS_UNION:
    def __init__(
        self,
        weights="weights/train_log_gmfss_union",
        scale=1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        trt=False,
        rife_weights=None,
        trt_cache_dir=None,
        cache_key="default",
    ):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.ifnet = IFNet(scale=scale).to(device).eval()

        if rife_weights is None:
            raise ValueError(
                "A RIFE 4.25 checkpoint path is required. Download it with "
                "python -m vsdrba and pass the path to flownet_v4.25.pkl."
            )
        rife_state = load_state_dict(rife_weights)
        load_result = self.ifnet.load_state_dict(rife_state, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(
                "Missing required RIFE 4.25 checkpoint parameters: "
                + ", ".join(load_result.missing_keys)
            )
        unexpected_other = [
            key for key in load_result.unexpected_keys
            if not key.startswith(("teacher.", "caltime."))
        ]
        if unexpected_other:
            raise RuntimeError(
                "Unexpected RIFE 4.25 checkpoint parameters: "
                + ", ".join(unexpected_other)
            )

        self.trt = bool(trt)
        if self.trt:
            try:
                _import_tensorrt_rtx()
            except Exception as exc:
                raise RuntimeError(
                    "TensorRT-RTX was requested but its runtime could not be imported. "
                    "Install the matching torch-tensorrt-rtx and tensorrt-rtx packages."
                ) from exc

            self.model.feat_ext.half()
            self.model.metricnet.half()
            self.model.fusionnet.half()
            self.model.flownet.half()
            self.ifnet.half()

            if trt_cache_dir is None:
                trt_cache_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "vsdrba", "tensorrt_rtx"
                )
            trt_cache_dir = os.path.realpath(os.path.expanduser(str(trt_cache_dir)))
            os.makedirs(trt_cache_dir, exist_ok=True)

            def cached(
                module,
                label,
                use_explicit_typing=False,
                free_module_after_compile=True,
            ):
                return _StrictTensorRTRTXModule(
                    module,
                    label,
                    cache_dir=trt_cache_dir,
                    cache_key=cache_key,
                    use_explicit_typing=use_explicit_typing,
                    free_module_after_compile=free_module_after_compile,
                )

            feat_ext_b1_mod = self.model.feat_ext
            feat_ext_b3_mod = copy.deepcopy(self.model.feat_ext)
            self.feat_ext_b1 = cached(feat_ext_b1_mod, "FeatureNet-b1")
            self.feat_ext_b3 = cached(feat_ext_b3_mod, "FeatureNet-b3")
            self.model.feat_ext = self.feat_ext_b1

            self.model.fusionnet = cached(self.model.fusionnet, "FusionNet")
            self.model.metricnet = cached(
                self.model.metricnet, "MetricNet-full", use_explicit_typing=True
            )
            self.flownet_bidir = cached(
                _GMFlowBidirWrapper(self.model.flownet), "GMFlow-bidir"
            )

            encode_b1_mod = self.ifnet.encode
            encode_b2_mod = copy.deepcopy(self.ifnet.encode)
            encode_b3_mod = copy.deepcopy(self.ifnet.encode)
            self.rife_encode_b1 = cached(encode_b1_mod, "RIFE-encoder-b1")
            self.rife_encode_b2 = cached(encode_b2_mod, "RIFE-encoder-b2")
            self.rife_encode_b3 = cached(encode_b3_mod, "RIFE-encoder-b3")
            self.ifnet.encode = self.rife_encode_b2

            self.ifnet_compiled = cached(
                self.ifnet, "RIFE-IFNet", use_explicit_typing=True
            )
        else:
            self.feat_ext_b1 = self.model.feat_ext
            self.feat_ext_b3 = None
            self.flownet_bidir = None
            self.rife_encode_b1 = self.ifnet.encode
            self.rife_encode_b2 = self.ifnet.encode
            self.rife_encode_b3 = self.ifnet.encode
            self.ifnet_compiled = self.ifnet

        self.scale = scale
        self.rife_weights = str(rife_weights)
        self._timestep_cache = {}

    def _extract_triplet_features(self, I0, I1, I2):
        """Extract FeatureNet pyramids for 3 frames in a single batched call."""
        if self.feat_ext_b3 is not None:
            feats = self.feat_ext_b3(torch.cat((I0, I1, I2), dim=0))
            return (
                [f[0:1] for f in feats],
                [f[1:2] for f in feats],
                [f[2:3] for f in feats],
            )
        feats = self.model.feat_ext(torch.cat((I0, I1, I2), dim=0))
        return (
            [f[0:1] for f in feats],
            [f[1:2] for f in feats],
            [f[2:3] for f in feats],
        )

    def _pad_for_rife(self, img: torch.Tensor) -> torch.Tensor:
        original_height, original_width = img.shape[-2:]
        align = max(64, int(round(64 / self.scale)))
        padded_height = ((original_height + align - 1) // align) * align
        padded_width = ((original_width + align - 1) // align) * align
        pad = (0, padded_width - original_width, 0, padded_height - original_height)
        if any(pad):
            return F.pad(img, pad, mode="replicate")
        return img

    def _encode_rife_frames(self, *half_imgs: torch.Tensor) -> list[torch.Tensor]:
        padded = [self._pad_for_rife(img) for img in half_imgs]
        batch = torch.cat(padded, dim=0)
        n = len(half_imgs)
        if n == 1:
            encoded = self.rife_encode_b1(batch)
        elif n == 2:
            encoded = self.rife_encode_b2(batch)
        elif n == 3:
            encoded = self.rife_encode_b3(batch)
        else:
            encoded = self.ifnet.encode(batch)
        return list(encoded.chunk(n, dim=0))

    def _get_timestep_tensor(self, reference: torch.Tensor, value: float) -> torch.Tensor:
        key = (reference.device, reference.shape[0], reference.shape[2], reference.shape[3], reference.dtype)
        tensor = self._timestep_cache.get(key)
        if tensor is None:
            tensor = reference[:, :1].new_empty((reference.shape[0], 1, reference.shape[2], reference.shape[3]))
            self._timestep_cache[key] = tensor
        return tensor.fill_(float(value))

    def _rife_inference(self, img0, img1, timestep, feat0=None, feat1=None):
        """Run the auxiliary RIFE network on dimensions safe for its five scales."""
        original_height, original_width = img0.shape[-2:]
        align = max(64, int(round(64 / self.scale)))
        padded_height = ((original_height + align - 1) // align) * align
        padded_width = ((original_width + align - 1) // align) * align
        pad = (0, padded_width - original_width, 0, padded_height - original_height)

        if any(pad):
            img0 = F.pad(img0, pad, mode="replicate")
            img1 = F.pad(img1, pad, mode="replicate")

        if not torch.is_tensor(timestep):
            timestep = self._get_timestep_tensor(img0, timestep)
        elif any(pad):
            timestep = F.pad(timestep, pad, mode="replicate")

        if feat0 is None or feat1 is None:
            encoded = self.rife_encode_b2(torch.cat((img0, img1), dim=0))
            feat0, feat1 = encoded.chunk(2, dim=0)

        prediction = self.ifnet_compiled(img0, img1, timestep, feat0, feat1)
        return prediction[:, :, :original_height, :original_width]

    def _run_pair_reuse(self, img0, img1, feat_ext0=None, feat_ext1=None, img0_half=None, img1_half=None):
        """Compute bidirectional flow and metric maps for one adjacent frame pair."""
        if feat_ext0 is None:
            feat_ext0 = list(self.feat_ext_b1(img0))
        if feat_ext1 is None:
            feat_ext1 = list(self.feat_ext_b1(img1))

        if self.flownet_bidir is None:
            return self.model.reuse(
                img0, img1, self.scale, feat_ext0=feat_ext0, feat_ext1=feat_ext1
            )

        if img0_half is None:
            img0_half = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        if img1_half is None:
            img1_half = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        target_height, target_width = img0_half.shape[-2:]

        if self.scale != 1.0:
            imgf0 = F.interpolate(img0_half, scale_factor=self.scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1_half, scale_factor=self.scale, mode="bilinear", align_corners=False)
        else:
            imgf0, imgf1 = img0_half, img1_half

        scaled_height, scaled_width = imgf0.shape[-2:]
        padded_height = ((scaled_height + 31) // 32) * 32
        padded_width = ((scaled_width + 31) // 32) * 32
        gmflow_pad = (0, padded_width - scaled_width, 0, padded_height - scaled_height)
        if any(gmflow_pad):
            imgf0 = F.pad(imgf0, gmflow_pad, mode="replicate")
            imgf1 = F.pad(imgf1, gmflow_pad, mode="replicate")

        flow_bidir = self.flownet_bidir(imgf0, imgf1)
        flow01, flow10 = flow_bidir.chunk(2, dim=0)
        flow01 = flow01[..., :scaled_height, :scaled_width]
        flow10 = flow10[..., :scaled_height, :scaled_width]

        if self.scale != 1.0:
            flow01 = F.interpolate(
                flow01, size=(target_height, target_width), mode="bilinear", align_corners=False
            ) / self.scale
            flow10 = F.interpolate(
                flow10, size=(target_height, target_width), mode="bilinear", align_corners=False
            ) / self.scale

        metric0, metric1 = self.model.metricnet(img0_half, img1_half, flow01, flow10)
        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(
        self, I0, I1, I2, ts, reuse=None, linear=False, return_current_reuse=False
    ):
        I0s = F.interpolate(I0, scale_factor=0.5, mode="bilinear", align_corners=False)
        I1s = F.interpolate(I1, scale_factor=0.5, mode="bilinear", align_corners=False)
        I2s = F.interpolate(I2, scale_factor=0.5, mode="bilinear", align_corners=False)

        if reuse is None:
            feat0, feat1, feat2 = self._extract_triplet_features(I0, I1, I2)
            rife_feat0, rife_feat1, rife_feat2 = self._encode_rife_frames(I0s, I1s, I2s)
            reuseI1I0 = list(
                self._run_pair_reuse(I1, I0, feat_ext0=feat1, feat_ext1=feat0, img0_half=I1s, img1_half=I0s)
            ) + [rife_feat1, rife_feat0]
            reuseI1I2 = list(
                self._run_pair_reuse(I1, I2, feat_ext0=feat1, feat_ext1=feat2, img0_half=I1s, img1_half=I2s)
            ) + [rife_feat1, rife_feat2]
        else:
            reuseI1I0 = list(reuse)
            feat2 = list(self.feat_ext_b1(I2))
            if len(reuseI1I0) >= 8:
                rife_feat1, rife_feat0 = reuseI1I0[6], reuseI1I0[7]
                (rife_feat2,) = self._encode_rife_frames(I2s)
            else:
                rife_feat0, rife_feat1, rife_feat2 = self._encode_rife_frames(I0s, I1s, I2s)
                reuseI1I0.extend([rife_feat1, rife_feat0])

            reuseI1I2 = list(
                self._run_pair_reuse(I1, I2, feat_ext0=reuseI1I0[4], feat_ext1=feat2, img0_half=I1s, img1_half=I2s)
            ) + [rife_feat1, rife_feat2]

        flow10, metric10 = reuseI1I0[0], reuseI1I0[2]
        flow12, metric12 = reuseI1I2[0], reuseI1I2[2]

        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            elif t == 2:
                output.append(I2)
            elif 0 < t < 1:
                t_sub = 1 - t
                drm_gmfss = calc_drm_gmfss(t_sub, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t_sub, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, I1s.shape[2:]) for k, v in drm_rife.items()}
                rife = self._rife_inference(
                    I1s, I0s, drm_rife["drm_t1_t01"], feat0=rife_feat1, feat1=rife_feat0
                )
                output.append(
                    self.model.inference(
                        I1, I0, reuseI1I0,
                        timestep0=drm_gmfss["drm1t_t01"],
                        timestep1=drm_gmfss["drm0t_t01"],
                        rife=rife,
                    )
                )
            elif 1 < t < 2:
                t_sub = t - 1
                drm_gmfss = calc_drm_gmfss(t_sub, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t_sub, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, I1s.shape[2:]) for k, v in drm_rife.items()}
                rife = self._rife_inference(
                    I1s, I2s, drm_rife["drm_t1_t12"], feat0=rife_feat1, feat1=rife_feat2
                )
                output.append(
                    self.model.inference(
                        I1, I2, reuseI1I2,
                        timestep0=drm_gmfss["drm1t_t12"],
                        timestep1=drm_gmfss["drm2t_t12"],
                        rife=rife,
                    )
                )

        next_reuse = [
            value
            for pair in zip(reuseI1I2[1::2], reuseI1I2[0::2])
            for value in pair
        ]
        if return_current_reuse:
            return output, next_reuse, reuseI1I0
        return output, next_reuse
