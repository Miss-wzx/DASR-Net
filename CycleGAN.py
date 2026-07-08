import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                3
            ),

            nn.InstanceNorm2d(channels),

            nn.GELU(),

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                3
            ),

            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):

        return x + self.block(x)

# LESS-CycleGAN Generator
class LESSGenerator(nn.Module):

    def __init__(
            self,
            in_channels=6,
            out_channels=6,
            ngf=64):

        super().__init__()

        # ====================================================
        # Head
        # ====================================================

        self.head = nn.Sequential(

            nn.ReflectionPad2d(3),

            nn.Conv2d(
                in_channels,
                ngf,
                7
            ),

            nn.InstanceNorm2d(ngf),

            nn.GELU()
        )

        # ====================================================
        # Down1
        # ====================================================

        self.down1 = nn.Sequential(

            nn.Conv2d(
                ngf,
                ngf * 2,
                3,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(
                ngf * 2
            ),

            nn.GELU()
        )

        # ====================================================
        # Down2
        # ====================================================

        self.down2 = nn.Sequential(

            nn.Conv2d(
                ngf * 2,
                ngf * 4,
                3,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(
                ngf * 4
            ),

            nn.GELU()
        )

        # ====================================================
        # 9 ResBlocks
        # ====================================================

        blocks = []

        for _ in range(9):
            blocks.append(
                ResBlock(
                    ngf * 4
                )
            )

        self.resblocks = nn.Sequential(
            *blocks
        )

        # ====================================================
        # Up1
        # ====================================================

        self.up1 = nn.Sequential(

            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(
                ngf * 2
            ),

            nn.GELU()
        )

        # ====================================================
        # Up2
        # ====================================================

        self.up2 = nn.Sequential(

            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(
                ngf
            ),

            nn.GELU()
        )

        # ====================================================
        # Output
        # ====================================================

        self.out_conv = nn.Sequential(

            nn.ReflectionPad2d(3),

            nn.Conv2d(
                ngf,
                out_channels,
                7
            )
        )

    def forward(self, x):

        x = self.head(x)

        x = self.down1(x)

        x = self.down2(x)

        x = self.resblocks(x)

        x = self.up1(x)

        x = self.up2(x)

        x = self.out_conv(x)

        # -----------------------------------
        # 32×32 → 80×80
        # -----------------------------------

        x = F.interpolate(
            x,
            size=(80, 80),
            mode='bicubic',
            align_corners=False
        )

        return x

if __name__ == "__main__":

    # LESS-CycleGAN Generator
    model = LESSGenerator()

    x = torch.randn(
        32,
        6,
        32,
        32
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
