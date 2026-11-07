"""GMFSS Union tensor inference and RIFE auxiliary adapter."""

import os

import torch
import torch.nn.functional as F

from .GMFSS import Model
from .drm import calc_drm_gmfss, calc_drm_rife_auxiliary
from .utils import load_state_dict, resize
from ..IFNet_HDv3_v4_26_heavy import IFNet


class GMFSS_UNION:
    def __init__(
        self,
        weights="weights/train_log_gmfss_union",
        scale=1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.ifnet = IFNet(scale=scale).to(device).eval()

        # The original DRBA checkpoint also contains teacher/caltime training
        # modules. They are not part of inference, but every inference parameter
        # must be present and load successfully.
        rife_state = load_state_dict(os.path.join(weights, "rife.pkl"))
        load_result = self.ifnet.load_state_dict(rife_state, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(
                "Missing required RIFE checkpoint parameters: "
                + ", ".join(load_result.missing_keys)
            )
        unexpected_other = [
            key for key in load_result.unexpected_keys
            if not key.startswith(("teacher.", "caltime."))
        ]
        if unexpected_other:
            raise RuntimeError(
                "Unexpected RIFE checkpoint parameters: "
                + ", ".join(unexpected_other)
            )
        if load_result.unexpected_keys:
            print(
                "RIFE checkpoint loaded; ignored "
                f"{len(load_result.unexpected_keys)} training-only teacher/caltime parameters."
            )

        self.scale = scale

    def _rife_inference(self, img0, img1, timestep):
        """Adapt GMFSS to VS-DRBA's optimized RIFE 4.26 Heavy forward API."""
        if not torch.is_tensor(timestep):
            timestep = img0[:, :1].new_full(
                (img0.shape[0], 1, img0.shape[2], img0.shape[3]), float(timestep)
            )
        feat0 = self.ifnet.encode(img0)
        feat1 = self.ifnet.encode(img1)
        return self.ifnet(img0, img1, timestep, feat0, feat1)

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
        reuseI1I2 = self.model.reuse(I1, I2, self.scale)

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
