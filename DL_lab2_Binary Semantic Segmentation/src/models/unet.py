# Implement your UNet model here

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Applies two consecutive Conv2d -> ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.upconv4(x5)
        x = self.center_crop_and_concat(x4, x)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = self.center_crop_and_concat(x3, x)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = self.center_crop_and_concat(x2, x)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = self.center_crop_and_concat(x1, x)
        x = self.dec1(x)

        return self.final_conv(x)

    def center_crop_and_concat(self, enc_feature, dec_feature):
        # Calculate cropping needed (for unpadded convs)
        _, _, h, w = dec_feature.size()
        enc_cropped = self.center_crop(enc_feature, h, w)
        return torch.cat([enc_cropped, dec_feature], dim=1)

    def center_crop(self, layer, target_height, target_width):
        _, _, h, w = layer.size()
        delta_h = h - target_height
        delta_w = w - target_width
        top = delta_h // 2
        left = delta_w // 2
        return layer[:, :, top:top+target_height, left:left+target_width]
