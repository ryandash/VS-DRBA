import torch
import torch.nn.functional as F

from .FeatureNet import FeatureNet
from .FusionNet import GridNet
from .MetricNet import MetricNet
from .gmflow.gmflow import GMFlow
from .softsplat import softsplat as warp
from .utils import load_state_dict


class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet(9, 64 * 2, 128 * 2, 192 * 2, 3)
        self.version = 3.9

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()
        return self

    def device(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)
        return self

    def load_model(self, path, rank=-1):
        checkpoint_names = (
            (self.flownet, "flownet.pkl"),
            (self.metricnet, "metric.pkl"),
            (self.feat_ext, "feat.pkl"),
            (self.fusionnet, "fusionnet.pkl"),
        )
        for network, filename in checkpoint_names:
            state_dict = load_state_dict(f"{path}/{filename}")
            network.load_state_dict(state_dict, strict=True)

    def reuse(self, img0, img1, scale, feat_ext0=None, feat_ext1=None, gmflow_features=None):
        if feat_ext0 is None and feat_ext1 is None:
            with torch.profiler.record_function("GMFSS/FeatureNet"):
                features = self.feat_ext(torch.cat((img0, img1), dim=0))
            feat_ext0 = [feature[:1] for feature in features]
            feat_ext1 = [feature[1:2] for feature in features]
        elif feat_ext0 is None:
            with torch.profiler.record_function("GMFSS/FeatureNet"):
                feat_ext0 = list(self.feat_ext(img0))
        elif feat_ext1 is None:
            with torch.profiler.record_function("GMFSS/FeatureNet"):
                feat_ext1 = list(self.feat_ext(img1))

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            imgf0, imgf1 = img0, img1

        target_height, target_width = img0.shape[-2:]
        scaled_height, scaled_width = imgf0.shape[-2:]
        padded_height = ((scaled_height + 31) // 32) * 32
        padded_width = ((scaled_width + 31) // 32) * 32
        gmflow_pad = (0, padded_width - scaled_width, 0, padded_height - scaled_height)
        if any(gmflow_pad):
            imgf0 = F.pad(imgf0, gmflow_pad, mode="replicate")
            imgf1 = F.pad(imgf1, gmflow_pad, mode="replicate")

        with torch.profiler.record_function("GMFSS/GMFlow"):
            if gmflow_features is None:
                flow_bidir = self.flownet(imgf0, imgf1, pred_bidir_flow=True)
            else:
                flow_bidir = self.flownet(
                    imgf0, imgf1, pred_bidir_flow=True,
                    feature0_list=gmflow_features[0],
                    feature1_list=gmflow_features[1],
                )
        flow01, flow10 = flow_bidir.chunk(2, dim=0)
        flow01 = flow01[..., :scaled_height, :scaled_width]
        flow10 = flow10[..., :scaled_height, :scaled_width]
        if scale != 1.0:
            flow01 = F.interpolate(
                flow01, size=(target_height, target_width),
                mode="bilinear", align_corners=False,
            ) / scale
            flow10 = F.interpolate(
                flow10, size=(target_height, target_width),
                mode="bilinear", align_corners=False,
            ) / scale

        with torch.profiler.record_function("GMFSS/MetricNet"):
            metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)
        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def inference(self, img0, img1, reuse_things, timestep0, timestep1, rife, enable_mask=True):
        flow01, metric0 = reuse_things[0], reuse_things[2]
        flow10, metric1 = reuse_things[1], reuse_things[3]
        feat11, feat12, feat13 = reuse_things[4]
        feat21, feat22, feat23 = reuse_things[5]

        F1t, F2t = timestep0 * flow01, timestep1 * flow10
        Z1t, Z2t = timestep0 * metric0, timestep1 * metric1

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)

        use_tensor_mask = isinstance(timestep0, torch.Tensor) and enable_mask

        # Fuse timestep0/timestep1 and coverage masks directly into the first SoftSplat call
        # alongside img0 + feat11 / img1 + feat21 to eliminate 2 full-res SoftSplat launches.
        if use_tensor_mask:
            t0_ch = timestep0.shape[1]
            t1_ch = timestep1.shape[1]
            ones0 = torch.ones_like(timestep0)
            ones1 = torch.ones_like(timestep1)

            warped0 = warp(
                torch.cat((img0, feat11, timestep0, ones0), dim=1),
                F1t, Z1t, strMode="soft"
            )
            I1t, feat1t1, timestep0, coverage0 = warped0.split(
                (img0.shape[1], feat11.shape[1], t0_ch, t0_ch), dim=1
            )

            warped1 = warp(
                torch.cat((img1, feat21, timestep1, ones1), dim=1),
                F2t, Z2t, strMode="soft"
            )
            I2t, feat2t1, timestep1, coverage1 = warped1.split(
                (img1.shape[1], feat21.shape[1], t1_ch, t1_ch), dim=1
            )
        else:
            warped0 = warp(torch.cat((img0, feat11), dim=1), F1t, Z1t, strMode="soft")
            I1t, feat1t1 = warped0.split((img0.shape[1], feat11.shape[1]), dim=1)
            warped1 = warp(torch.cat((img1, feat21), dim=1), F2t, Z2t, strMode="soft")
            I2t, feat2t1 = warped1.split((img1.shape[1], feat21.shape[1]), dim=1)

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode="soft")
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode="soft")

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode="soft")
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode="soft")

        if use_tensor_mask:
            gaps0 = coverage0 < 0.999
            gaps1 = coverage1 < 0.999
            invalid_mask = torch.logical_or(gaps0, gaps1)
            timestep0 = timestep0.masked_fill(invalid_mask, 1)
            timestep1 = timestep1.masked_fill(invalid_mask, 1)

            def get_mask(channels, scale=1.0):
                t0, t1 = timestep0, timestep1
                if scale != 1.0:
                    t0 = F.interpolate(t0, scale_factor=scale, mode="bilinear", align_corners=False)
                    t1 = F.interpolate(t1, scale_factor=scale, mode="bilinear", align_corners=False)
                return (t0 / t1 > 25).expand(-1, channels, -1, -1), (t1 / t0 > 25).expand(-1, channels, -1, -1)

            mask0, mask1 = get_mask(3)
            I1t, I2t = torch.where(mask0, I2t, I1t), torch.where(mask1, I1t, I2t)
            mask0, mask1 = get_mask(64)
            feat1t1, feat2t1 = torch.where(mask0, feat2t1, feat1t1), torch.where(mask1, feat1t1, feat2t1)
            mask0, mask1 = get_mask(128, 0.5)
            feat1t2, feat2t2 = torch.where(mask0, feat2t2, feat1t2), torch.where(mask1, feat1t2, feat2t2)
            mask0, mask1 = get_mask(192, 0.25)
            feat1t3, feat2t3 = torch.where(mask0, feat2t3, feat1t3), torch.where(mask1, feat1t3, feat2t3)

        with torch.profiler.record_function("GMFSS/FusionNet"):
            out = self.fusionnet(
                torch.cat([I1t, rife, I2t], dim=1),
                torch.cat([feat1t1, feat2t1], dim=1),
                torch.cat([feat1t2, feat2t2], dim=1),
                torch.cat([feat1t3, feat2t3], dim=1),
            )
        return torch.clamp(out, 0, 1)
