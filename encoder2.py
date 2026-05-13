import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            Swish(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = Swish()

    def forward(self, x):
        return self.activation(x + self.block(x))

# Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, level=3, max_channels=256):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.level = level
        self.max_channels = max_channels

        self.initial = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            Swish()
        ])

        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(level):

            out_ch = in_ch if i == 0 else min(in_ch * 2, max_channels)
            
            block = nn.ModuleDict({
                'down': nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                'res': ResidualBlock(out_ch)
            })
            self.blocks.append(block)
            in_ch = out_ch

        final_ch = min(in_ch * 2, max_channels)
        self.final_conv = nn.Conv2d(in_ch, final_ch, kernel_size=3, padding=1)
        self.final_res = ResidualBlock(final_ch)

    def forward(self, x):
        fea = []
        for layer in self.initial:
            x = layer(x)
        for block in self.blocks:
            x = block['down'](x)
            x = block['res'](x)
            fea.append(x)
        x = self.final_conv(x)
        x = self.final_res(x)
        fea.append(x)
        return fea