"""GMFSS Union tensor inference and RIFE auxiliary adapter."""

import os
import sys

import torch
import torch.nn.functional as F

from .GMFSS import Model
from .drm import calc_drm_gmfss, calc_drm_rife_auxiliary
from .utils import load_state_dict, resize
from ..IFNet_HDv3_v4_25 import IFNet


class _StrictTensorRTRTXModule(torch.nn.Module):
    """Lazily export and compile one fixed-shape module with full RTX coverage."""

    def __init__(self, module, label):
        super().__init__()
        self.module = module
        self.label = label
        self.compiled = None

    def forward(self, *inputs):
        if self.compiled is None:
            try:
                import torch_tensorrt

                exported = torch.export.export(self.module, tuple(inputs))
                self.compiled = torch_tensorrt.dynamo.compile(
                    exported,
                    inputs=list(inputs),
                    enabled_precisions={torch.float16},
                    optimization_level=3,
                    min_block_size=1,
                    require_full_compilation=True,
                    pass_through_build_failures=True,
                ).eval()
            except Exception as exc:
                raise RuntimeError(
                    f"TensorRT-RTX could not fully compile {self.label}; "
                    f"no PyTorch fallback is permitted: {exc}"
                ) from exc
        return self.compiled(*inputs)


class _MetricConvHead(torch.nn.Module):
    """Module adapter for MetricNet's convolution-only head."""

    def __init__(self, metricnet):
        super().__init__()
        self.metric_in = metricnet.metric_in
        self.metric_net1 = metricnet.metric_net1
        self.metric_net2 = metricnet.metric_net2
        self.metric_net3 = metricnet.metric_net3
        self.metric_out = metricnet.metric_out

    def forward(self, feat):
        feat = self.metric_in(feat)
        feat = self.metric_net1(feat) + feat
        feat = self.metric_net2(feat) + feat
        feat = self.metric_net3(feat) + feat
        metric = self.metric_out(feat) * 10
        return metric[:, :1], metric[:, 1:2]


class GMFSS_UNION:
    def __init__(
        self,
        weights="weights/train_log_gmfss_union",
        scale=1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        trt=False,
        rife_weights=None,
    ):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.ifnet = IFNet(scale=scale).to(device).eval()

        # Load the standalone RIFE 4.25 checkpoint. The 4.25 encoder emits
        # 4 feature channels (the 4.26 Heavy encoder emits 16), so its weights
        # are architecturally incompatible with the old GMFSS heavy checkpoint.
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
        if load_result.unexpected_keys:
            print(
                "RIFE checkpoint loaded; ignored "
                f"{len(load_result.unexpected_keys)} training-only teacher/caltime parameters.",
                file=sys.stderr,
            )


        self.trt = bool(trt)
        if self.trt:
            # TensorRT-RTX is a separate runtime from standard TensorRT.
            # Import failures are fatal: never silently disable acceleration.
            try:
                import tensorrt_rtx
                import torch_tensorrt
            except Exception as exc:
                raise RuntimeError(
                    "TensorRT-RTX was requested but its runtime could not be imported. "
                    "Install the matching torch-tensorrt-rtx and tensorrt-rtx packages."
                ) from exc

            self.model.feat_ext.half()
            self.model.metricnet.half()
            self.model.fusionnet.half()
            self.ifnet.half()

            # Compile only deliberately selected complete submodules.
            # SoftSplat, DRM, GMFlow transformer/matching, and RIFE warping
            # remain outside the engine boundary by design. Each selected
            # module must compile fully; unsupported operators are fatal.
            # Modules with custom warping or non-Tensor arguments stay outside
            # the engine boundary intentionally rather than being partitioned
            # into hidden eager fallback regions.
            self.model.feat_ext = _StrictTensorRTRTXModule(
                self.model.feat_ext, "FeatureNet"
            )
            self.model.fusionnet = _StrictTensorRTRTXModule(
                self.model.fusionnet, "FusionNet"
            )
            self.model.metricnet.forward_conv_rtx = _StrictTensorRTRTXModule(
                _MetricConvHead(self.model.metricnet), "MetricNet convolution head"
            )
            self.model.flownet.backbone = _StrictTensorRTRTXModule(
                self.model.flownet.backbone, "GMFlow CNN backbone"
            )
            self.ifnet.encode = _StrictTensorRTRTXModule(
                self.ifnet.encode, "RIFE encoder"
            )

            print(
                f"TensorRT-RTX {tensorrt_rtx.__version__} enabled for FeatureNet, "
                "MetricNet convolution head, FusionNet, GMFlow CNN backbone, "
                "and RIFE encoder. Full-engine compilation is enforced; unsupported modules stay outside the engine boundary.",
                file=sys.stderr,
            )

        self.scale = scale
        self.rife_weights = str(rife_weights)

    def _rife_inference(self, img0, img1, timestep):
        """Run the auxiliary RIFE network on dimensions safe for its five scales.

        The RIFE Heavy IFBlocks use two stride-2 convolutions at each pyramid
        level. Padding the half-resolution inputs to multiples of 64 prevents
        odd intermediate feature sizes from rounding up and producing a flow
        tensor larger than the corresponding warp grid.
        """
        original_height, original_width = img0.shape[-2:]
        padded_height = ((original_height + 63) // 64) * 64
        padded_width = ((original_width + 63) // 64) * 64
        pad = (0, padded_width - original_width, 0, padded_height - original_height)

        if any(pad):
            img0 = F.pad(img0, pad, mode="replicate")
            img1 = F.pad(img1, pad, mode="replicate")

        if not torch.is_tensor(timestep):
            timestep = img0[:, :1].new_full(
                (img0.shape[0], 1, img0.shape[2], img0.shape[3]), float(timestep)
            )
        elif any(pad):
            timestep = F.pad(timestep, pad, mode="replicate")

        with torch.profiler.record_function("GMFSS/RIFE_Encoder"):
            feat0 = self.ifnet.encode(img0)
            feat1 = self.ifnet.encode(img1)
        with torch.profiler.record_function("GMFSS/RIFE_IFNet"):
            prediction = self.ifnet(img0, img1, timestep, feat0, feat1)
        return prediction[:, :, :original_height, :original_width]

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, I0, I1, ts):
        reuse = self.model.reuse(I0, I1, self.scale)
        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            else:
                I0s = F.interpolate(I0, scale_factor=0.5, mode="bilinear", align_corners=False)
                I1s = F.interpolate(I1, scale_factor=0.5, mode="bilinear", align_corners=False)
                rife = self._rife_inference(I0s, I1s, t)
                output.append(
                    self.model.inference(
                        I0, I1, reuse, timestep0=t, timestep1=1 - t, rife=rife
                    )
                )
        return output

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, I0, I1, I2, ts, reuse=None, linear=False):
        reuseI1I0 = self.model.reuse(I1, I0, self.scale) if reuse is None else reuse
        reuseI1I2 = self.model.reuse(
            I1, I2, self.scale, feat_ext0=reuseI1I0[4]
        )

        flow10, metric10 = reuseI1I0[0], reuseI1I0[2]
        flow12, metric12 = reuseI1I2[0], reuseI1I2[2]
        I0s, I1s, I2s = [
            F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            for x in (I0, I1, I2)
        ]

        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            elif t == 2:
                output.append(I2)
            elif 0 < t < 1:
                t = 1 - t
                drm_gmfss = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, I0s.shape[2:]) for k, v in drm_rife.items()}
                rife = self._rife_inference(I1s, I0s, drm_rife["drm_t1_t01"])
                output.append(
                    self.model.inference(
                        I1, I0, reuseI1I0,
                        timestep0=drm_gmfss["drm1t_t01"],
                        timestep1=drm_gmfss["drm0t_t01"],
                        rife=rife,
                    )
                )
            elif 1 < t < 2:
                t = t - 1
                drm_gmfss = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear)
                drm_rife = {k: resize(v, I0s.shape[2:]) for k, v in drm_rife.items()}
                rife = self._rife_inference(I1s, I2s, drm_rife["drm_t1_t12"])
                output.append(
                    self.model.inference(
                        I1, I2, reuseI1I2,
                        timestep0=drm_gmfss["drm1t_t12"],
                        timestep1=drm_gmfss["drm2t_t12"],
                        rife=rife,
                    )
                )

        reuse = [
            value
            for pair in zip(reuseI1I2[1::2], reuseI1I2[0::2])
            for value in pair
        ]
        return output, reuse
