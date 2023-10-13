import torch
import torch.nn as nn


class UNet(nn.Module):
    '''
    UNet model with diffusion included.

    '''
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.first_out_channels = first_out_channels

        # ### FEATURE ENCODER ####
        # self.first_fe = ConvBlock(in_channels-1, first_out_channels)
        # in_channels = first_out_channels
        # self.down1 = Down(in_channels, 2 * in_channels)     # 16 -> 32
        # self.down2 = Down(2 * in_channels, 4 * in_channels) # 32 -> 64
        # self.down3 = Down(4 * in_channels, 8 * in_channels) # 64 -> 128

        ### FEATURE ENCODER ####
        self.first_fe = ConvBlock(in_channels-1, first_out_channels)

        ### DENOISING U-NET ####
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)     # 16 -> 32
        self.down2 = Down(2 * in_channels, 4 * in_channels) # 32 -> 64
        self.down3 = Down(4 * in_channels, 8 * in_channels) # 64 -> 128
        self.up1   = Up(8 * in_channels, 4 * in_channels)   # 128 -> 64
        self.up2   = Up(4 * in_channels, 2 * in_channels)   # 64 -> 32
        self.up3   = Up(2 * in_channels, in_channels)       # 32 -> 16
        # print(out_channels)
        # self.prefinal = nn.Conv3d(in_channels, 0.5 * in_channels, 1)
        # self.final = nn.Conv3d(0.5 * in_channels, num_classes, 1)
        self.prefinal = nn.Conv3d(in_channels, first_out_channels, 1)
        self.final = nn.Conv3d(first_out_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image):
        # x_diff = image + int(torch.normal(torch.tensor(0.), torch.tensor(10.)))
        # print('x_diff: ', x_diff.shape)
        # concat = torch.cat([image, x_diff], dim=1)
        # print('concat: ', concat.shape)

        # ### DENOISING U-NET ###
        # # Conv
        # den_co = self.first(concat)
        # print('den_co: ', den_co.shape)


        ### FEATURE ENCODER ####
        # Convolving and downsampling the original image
        # From original FracNet vars to this version:
        # x1 -> x_co
        # x2 -> x_d1
        # x3 -> x_d2
        # x4 -> x_d3
        x_co = self.first_fe(image)    # 2 (Conv 3x3x3, BN, LReLU)
        x_d1 = self.down1(x_co)     # 16 -> 32
        x_d2 = self.down2(x_d1)     # 32 -> 64
        x_d3 = self.down3(x_d2)     # 64 -> 128

        ####### NOISE #########
        x_diff = image + int(torch.normal(torch.tensor(0.), torch.tensor(20.)))

        # CHANNEL-WISE CONCAT #
        concat = torch.cat([image, x_diff], dim=1)

        ### DENOISING U-NET ###
        # Conv
        den_co = self.first(concat)
        den_co = den_co + x_co

        # First downsampling step
        den_d1 = self.down1(den_co)
        den_d1 = den_d1 + x_d1

        # Second downsampling step
        den_d2 = self.down2(den_d1)
        den_d2 = den_d2 + x_d2

        # Third downsampling step
        den_d3 = self.down3(den_d2)
        den_d3 = den_d3 + x_d3

        # Upsampling
        den_u3  = self.up1(den_d3, den_d2)   # 128 -> 64
        den_u2  = self.up2(den_u3, den_d1)   # 64 -> 32
        den_u1  = self.up3(den_u2, den_co)    # 32 -> 16
        # print(den_u1.shape)
        conv_pre = self.prefinal(den_u1)
        # print(conv_pre.shape)
        # raise NotImplementedError

        logits  = self.final(conv_pre)      # Conv 3x3x3
        print(logits.shape)
        return logits

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

