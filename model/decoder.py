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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, proto_dim=0, prior_dim=0):
        super().__init__()

        self.proto_dim = proto_dim
        self.prior_dim = prior_dim
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2, prototype=None, prior=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        parts = [x2, x1]
        if self.proto_dim > 0:
            assert prototype is not None and prototype.shape[1] == self.proto_dim
            parts.append(prototype.expand(-1, -1, x2.size()[2], x2.size()[3]))
        if self.prior_dim > 0:
            assert prior is not None and prior.shape[1] == 1
            parts.append(F.interpolate(prior, size=x2.shape[-2:], mode='bilinear', align_corners=False))

        x = torch.cat(parts, dim=1)
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


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

        self.rfb1 = RFB_modified(x1_ch, 64)
        self.rfb2 = RFB_modified(x2_ch, 64)
        self.rfb3 = RFB_modified(x3_ch, 64)
        self.rfb4 = RFB_modified(x4_ch, 64)

        self.up1 = Up(128 + self.proto_dim + self.prior_dim, 64, self.proto_dim, self.prior_dim)
        self.up2 = Up(128 + self.proto_dim + self.prior_dim, 64, self.proto_dim, self.prior_dim)
        self.up3 = Up(128 + self.proto_dim + self.prior_dim, 64, self.proto_dim, self.prior_dim)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features, prototype=None, prior=None):
        x1 = features['0']
        x2 = features['1']
        x3 = features['2']
        x4 = features['3']

        assert x1.shape[1] == self.x1_ch
        assert x2.shape[1] == self.x2_ch
        assert x3.shape[1] == self.x3_ch
        assert x4.shape[1] == self.x4_ch

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        x = self.up1(x4, x3, prototype, prior)
        x = self.up2(x, x2, prototype, prior)
        x = self.up3(x, x1, prototype, prior)

        return F.interpolate(self.head(x), scale_factor=4, mode='bilinear', align_corners=False)
