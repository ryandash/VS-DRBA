import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmflow.geometry import forward_backward_consistency_check


def backwarp(tenIn: torch.Tensor, tenflow: torch.Tensor) -> torch.Tensor:
    """Stateless bilinear backwarp compatible with torch.export and TensorRT-RTX."""
    batch, _, height, width = tenflow.shape
    tenHor = (
        torch.linspace(start=-1.0, end=1.0, steps=width, dtype=tenflow.dtype, device=tenflow.device)
        .view(1, 1, 1, width)
        .expand(batch, 1, height, width)
    )
    tenVer = (
        torch.linspace(start=-1.0, end=1.0, steps=height, dtype=tenflow.dtype, device=tenflow.device)
        .view(1, 1, height, 1)
        .expand(batch, 1, height, width)
    )
    base_grid = torch.cat([tenHor, tenVer], dim=1)

    norm_flow = torch.cat(
        [
            tenflow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
            tenflow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )

    sample_grid = (base_grid + norm_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenIn,
        grid=sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_in = nn.Conv2d(14, 64, 3, 1, 1)
        self.metric_net1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.metric_net2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.metric_net3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.metric_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(64, 2, 3, 1, 1),
            nn.Tanh(),
        )
        self.forward_conv_rtx = None

    def forward(self, img0, img1, flow01, flow10):
        metric0 = F.l1_loss(img0, backwarp(img1, flow01), reduction="none").mean([1], True)
        metric1 = F.l1_loss(img1, backwarp(img0, flow10), reduction="none").mean([1], True)

        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        flow01_norm = torch.cat(
            [
                flow01[:, 0:1, :, :] / ((flow01.shape[3] - 1.0) / 2.0),
                flow01[:, 1:2, :, :] / ((flow01.shape[2] - 1.0) / 2.0),
            ],
            dim=1,
        )
        flow10_norm = torch.cat(
            [
                flow10[:, 0:1, :, :] / ((flow10.shape[3] - 1.0) / 2.0),
                flow10[:, 1:2, :, :] / ((flow10.shape[2] - 1.0) / 2.0),
            ],
            dim=1,
        )

        img = torch.cat((img0, img1), 1)
        metric = torch.cat((-metric0, -metric1), 1)
        flow = torch.cat((flow01_norm, flow10_norm), 1)
        occ = torch.cat((fwd_occ.unsqueeze(1).to(img0.dtype), bwd_occ.unsqueeze(1).to(img0.dtype)), 1)

        features = torch.cat((img, metric, flow, occ), 1)
        if self.forward_conv_rtx is not None:
            return self.forward_conv_rtx(features)
        return self.forward_conv(features)

    def forward_conv(self, feat):
        """Convolution-only MetricNet head."""
        feat = self.metric_in(feat)
        feat = self.metric_net1(feat) + feat
        feat = self.metric_net2(feat) + feat
        feat = self.metric_net3(feat) + feat
        metric = self.metric_out(feat) * 10
        return metric[:, :1], metric[:, 1:2]
