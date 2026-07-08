import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Dense Layer
# ============================================================

class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, growth_rate, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)


# ============================================================
# Dense Block
# ============================================================

class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, n_layers=4):
        super().__init__()

        layers = []
        channels = in_channels

        for _ in range(n_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


# ============================================================
# Transition Down
# ============================================================

class TransitionDown(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.down(x)


# ============================================================
# Transition Up
# ============================================================

class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        return self.up(x)


# ============================================================
# FD-UNet (for SAM SR task)
# Input : [B, 6, 32, 32]
# Output: [B, 6, 80, 80]
# ============================================================

class FDUNet(nn.Module):

    def __init__(self,
                 in_channels=6,
                 out_channels=6,
                 growth_rate=32):

        super().__init__()

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Conv2d(
            in_channels,
            48,
            3,
            1,
            1
        )

        # -------------------------
        # Encoder
        # -------------------------
        self.db1 = DenseBlock(48, growth_rate, 4)
        ch1 = self.db1.out_channels

        self.td1 = TransitionDown(ch1)

        self.db2 = DenseBlock(ch1, growth_rate, 4)
        ch2 = self.db2.out_channels

        self.td2 = TransitionDown(ch2)

        self.db3 = DenseBlock(ch2, growth_rate, 4)
        ch3 = self.db3.out_channels

        self.td3 = TransitionDown(ch3)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = DenseBlock(ch3, growth_rate, 4)
        bottleneck_ch = self.bottleneck.out_channels

        # -------------------------
        # Decoder
        # -------------------------
        self.tu3 = TransitionUp(bottleneck_ch, ch3)
        self.db_up3 = DenseBlock(ch3 * 2, growth_rate, 4)
        up3_ch = self.db_up3.out_channels

        self.tu2 = TransitionUp(up3_ch, ch2)
        self.db_up2 = DenseBlock(ch2 * 2, growth_rate, 4)
        up2_ch = self.db_up2.out_channels

        self.tu1 = TransitionUp(up2_ch, ch1)
        self.db_up1 = DenseBlock(ch1 * 2, growth_rate, 4)
        up1_ch = self.db_up1.out_channels

        # -------------------------
        # SR Head (IMPORTANT)
        # -------------------------
        up1_ch = up2_ch
        self.sr_head = nn.Sequential(
            nn.Conv2d(up1_ch, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, 1, 1)
        )

    def forward(self, x):

        # Encoder
        x0 = self.stem(x)

        x1 = self.db1(x0)
        d1 = self.td1(x1)

        x2 = self.db2(d1)
        d2 = self.td2(x2)

        x3 = self.db3(d2)
        d3 = self.td3(x3)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder
        u3 = self.tu3(b)
        u3 = torch.cat([u3, x3], dim=1)
        # u3 = u3 + 0 * x3
        u3 = self.db_up3(u3)

        u2 = self.tu2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.db_up2(u2)

        u1 = self.tu1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.db_up1(u1)

        # Output
        # out = self.sr_head(u1)
        out = self.sr_head(u2)

        # ---- resize to 80×80 (FD-UNet baseline convention) ----
        out = F.interpolate(
            out,
            size=(80, 80),
            mode='bicubic',
            align_corners=False
        )

        return out

if __name__ == "__main__":

    model = FDUNet(
        in_channels=6,
        out_channels=6
    )

    x = torch.randn(32, 6, 32, 32)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
