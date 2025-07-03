# Implement your ResNet34_UNet model here

import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        base_model = models.resnet34(weights=None)  # 不使用預訓練權重
        self.base_layers = list(base_model.children())

        self.input_layer = nn.Sequential(*self.base_layers[:3])     # conv1 + bn1 + relu
        self.input_pool = self.base_layers[3]                       # maxpool
        self.encoder1 = self.base_layers[4]                         # layer1
        self.encoder2 = self.base_layers[5]                         # layer2
        self.encoder3 = self.base_layers[6]                         # layer3
        self.encoder4 = self.base_layers[7]                         # layer4

        self.center = DoubleConv(512, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def center_crop(self, layer, target_height, target_width):
        _, _, h, w = layer.size()
        delta_h = h - target_height
        delta_w = w - target_width
        top = delta_h // 2
        left = delta_w // 2
        return layer[:, :, top:top+target_height, left:left+target_width]

    def forward(self, x):
        x0 = self.input_layer(x)      # H/2
        x1 = self.input_pool(x0)      # H/4
        x2 = self.encoder1(x1)        # H/4
        x3 = self.encoder2(x2)        # H/8
        x4 = self.encoder3(x3)        # H/16
        x5 = self.encoder4(x4)        # H/32

        center = self.center(x5)      # Bottleneck

        d4 = self.up4(center)
        x4_crop = self.center_crop(x4, d4.size(2), d4.size(3))
        d4 = self.dec4(torch.cat([d4, x4_crop], dim=1))

        d3 = self.up3(d4)
        x3_crop = self.center_crop(x3, d3.size(2), d3.size(3))
        d3 = self.dec3(torch.cat([d3, x3_crop], dim=1))

        d2 = self.up2(d3)
        x2_crop = self.center_crop(x2, d2.size(2), d2.size(3))
        d2 = self.dec2(torch.cat([d2, x2_crop], dim=1))

        d1 = self.up1(d2)
        x0_crop = self.center_crop(x0, d1.size(2), d1.size(3))
        d1 = self.dec1(torch.cat([d1, x0_crop], dim=1))

        out = self.final(d1)

        out = nn.functional.interpolate(
            out,
            size=(x.size(2), x.size(3)),
            mode='bilinear',
            align_corners=False
        )
        return out