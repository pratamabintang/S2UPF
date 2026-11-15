import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Decoder(nn.Module):
    def __init__(
        self,
        x1_ch,
        x2_ch,
        x3_ch,
        x4_ch,
        proto_dim=64,
        use_prototype=True,
        use_prior=True
    ):
        super(Decoder, self).__init__()
        self.x1_ch = x1_ch
        self.x2_ch = x2_ch
        self.x3_ch = x3_ch
        self.x4_ch = x4_ch
        self.use_prototype = use_prototype
        self.use_prior = use_prior
        self.proto_dim = proto_dim if use_prototype else 0
        self.prior_dim = 1 if use_prior else 0

        self.bottleneck = DoubleConv(x4_ch, 256)
        self.dec16 = DoubleConv(256 + x3_ch + self.proto_dim + self.prior_dim, 256)
        self.dec8 = DoubleConv(256 + x2_ch + self.proto_dim + self.prior_dim, 128)
        self.dec4 = DoubleConv(128 + x1_ch + self.proto_dim + self.prior_dim, 64)
        self.seg_head = nn.Conv2d(64, 1, kernel_size=1)

    def _merge(self, x, skip, prototype, prior):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        parts = [x, skip]
        if self.use_prototype:
            assert prototype is not None and prototype.shape[1] == self.proto_dim
            parts.append(
                prototype.expand(-1, -1, skip.shape[-2], skip.shape[-1])
            )
        if self.use_prior:
            assert prior is not None and prior.shape[1] == 1
            parts.append(
                F.interpolate(
                    prior,
                    size=skip.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            )
        return torch.cat(parts, dim=1)

    def forward(self, features, prototype=None, prior=None):
        x1 = features['0']
        x2 = features['1']
        x3 = features['2']
        x4 = features['3']

        assert x1.shape[1] == self.x1_ch
        assert x2.shape[1] == self.x2_ch
        assert x3.shape[1] == self.x3_ch
        assert x4.shape[1] == self.x4_ch

        x = self.bottleneck(x4)
        x = self.dec16(self._merge(x, x3, prototype, prior))
        x = self.dec8(self._merge(x, x2, prototype, prior))
        x = self.dec4(self._merge(x, x1, prototype, prior))

        return self.seg_head(x)
