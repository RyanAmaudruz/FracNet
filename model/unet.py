import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.first_out_channels = first_out_channels
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)     # 16 -> 32
        self.down2 = Down(2 * in_channels, 4 * in_channels) # 32 -> 64
        self.down3 = Down(4 * in_channels, 8 * in_channels) # 64 -> 128
        self.up1   = Up(8 * in_channels, 4 * in_channels)   # 128 -> 64
        self.up2   = Up(4 * in_channels, 2 * in_channels)   # 64 -> 32
        self.up3   = Up(2 * in_channels, in_channels)       # 32 -> 16
        self.final = nn.Conv3d(in_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.first(x)      # 2 (Conv 3x3x3, BN, LReLU)
        x2 = self.down1(x1)     # 16 -> 32
        x3 = self.down2(x2)     # 32 -> 64
        x4 = self.down3(x3)     # 64 -> 128
        x  = self.up1(x4, x3)   # 128 -> 64
        x  = self.up2(x, x2)    # 64 -> 32
        x  = self.up3(x, x1)    # 32 -> 16
        x  = self.final(x)      # Conv 3x3x3
        return x

    def __str__(self):
        return f'unet_in-{self.in_channels}_out-{self.num_classes}_fout-{self.first_out_channels}'


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x
