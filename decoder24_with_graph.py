
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import torch as th
import numpy as np
import json
import uuid
import os 

class FeatureReducer3(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)  # (B, hidden_channels)


class WeightedConvPool3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,), num_convs=4):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList()

        for ks in kernel_sizes:
            for _ in range(num_convs):
                padding = ks // 2
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=padding))

        self.total_convs = len(self.convs)

        self.feature_reducer = FeatureReducer3(in_channels, hidden_channels=256)

        self.fc_weight = nn.Linear(256, self.num_convs * out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.feature_reducer(x)  # (B, hidden)
        weights = self.fc_weight(feat).view(B, self.num_convs, self.out_channels, 1, 1)  # (B, N, C, 1, 1)

        outputs = [conv(x) for conv in self.convs]  # list of (B, C, H, W)
        stacked = torch.stack(outputs, dim=1)  # (B, N, C, H, W)

        # weights = weights.view(B, self.total_convs, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)
        return out


class FeatureReducer5(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)  # (B, hidden_channels)


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_candidates = nn.ModuleList([
            WeightedConvPool3(in_channels, out_channels),
        ])

        self.act_candidates = nn.ModuleList([
            nn.LeakyReLU(0.2),
            nn.GELU(),
            nn.SiLU()
        ])

        self.norm_candidates = nn.ModuleList([
            nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
        ])

        self.act_name_map = {
            'LeakyReLU': self.act_candidates[0],
            'GELU': self.act_candidates[1],
            'SiLU': self.act_candidates[2]
        }
        self.norm_name_map = {
            'BatchNorm2d': self.norm_candidates[0],
            'InstanceNorm2d': self.norm_candidates[1],
            'GroupNorm': self.norm_candidates[2]
        }
        self.conv_name_map = {
            'WeightedConvPool3': self.conv_candidates[0]
        }

    def forward(self, x, arch_spec=None):
        if arch_spec is None:
            conv_op = random.choice(self.conv_candidates)
            act_op = random.choice(self.act_candidates)
            norm_op = random.choice(self.norm_candidates)

            ops = [('conv', conv_op), ('act', act_op), ('norm', norm_op)]
            random.shuffle(ops)

        else:
            ops = []
            for layer_info in arch_spec:
                typ = layer_info['type']
                op_name = layer_info['op']
                if typ == 'conv':
                    ops.append((typ, self.conv_name_map[op_name]))
                elif typ == 'act':
                    ops.append((typ, self.act_name_map[op_name]))
                elif typ == 'norm':
                    ops.append((typ, self.norm_name_map[op_name]))
                else:
                    continue

        trace = []
        for typ, op in ops:
            x = op(x)
            trace.append({'type': typ, 'op': op.__class__.__name__})
            
        return x, trace


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_subblocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subblocks = num_subblocks
        self.subblocks = nn.ModuleList([SubBlock(out_channels, out_channels) for _ in range(num_subblocks)])

        self.upsample_pixelshuffle = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2))        
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.upsample_methods = [
            'bilinear',
            'nearest',
            'bicubic',
            'pixelshuffle']

    def forward(self, x, base_index, arch_spec=None):
        x = self.conv_up(x)
        method = None
        if arch_spec is None:
            method = random.choice(self.upsample_methods)
        else:
            method = arch_spec[0]['op']

        if method == 'bilinear':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif method == 'nearest':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        elif method == 'bicubic':
            x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        elif method == 'pixelshuffle':
            x = self.upsample_pixelshuffle(x)

        arch = []
        arch.append({
            'index': base_index,
            'type': 'Upsample',
            'op': method
        })

        res_x = x
        num_active = None
        if arch_spec is None:
            num_active = random.randint(1, self.num_subblocks)
        else:
            num_active = 0
            for i in range(self.num_subblocks):
                subblock_layers = arch_spec[1 + i*3 : 1 + (i+1)*3]
                if any(l['type'] is not None for l in subblock_layers):
                    num_active += 1

        for i in range(self.num_subblocks):
            layer_idx_base = base_index + 1 + i * 3
            if i < num_active:
                subblock_spec = None
                if arch_spec is not None:
                    subblock_spec = arch_spec[1 + i*3 : 1 + (i+1)*3]
                x, trace = self.subblocks[i](x, arch_spec=subblock_spec)
                for j, layer_info in enumerate(trace):
                    arch.append({
                        'index': layer_idx_base + j,
                        'type': layer_info['type'],
                        'op': layer_info['op']
                    })
            else:
                for j in range(3):
                    arch.append({
                        'index': layer_idx_base + j,
                        'type': None,
                        'op': None
                    })

        return x + res_x, arch



# 512(32)-256(64)-128(128)-64(256)
class Decoder(nn.Module):
    def __init__(self, in_channels=256, base_channels=128, levels=3, num_subblocks=2):
        super().__init__()
        self.in_channels_list = [256, 128, 64]  # Corresponding to Feature 3, 2, 1, 0
        self.out_channels_list = [128, 64, 32]  # Corresponding to Feature 3, 2, 1, 0

        self.blocks = nn.ModuleList()
        for i in range(levels):
            in_channels = self.in_channels_list[i]
            out_channels = self.out_channels_list[i]
            self.blocks.append(UpsampleBlock(in_channels, out_channels, num_subblocks))  #(32-64)256/128  (64-128)128/64  (128-256)64/32
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, features, arch_json=None):
        valid_indices = [0, 1, 3]
        start_idx = random.choice(valid_indices) if arch_json is None else None

        if arch_json is not None:
            first_block_index = arch_json[0]['index']
            if first_block_index == 1:
                start_idx = 3
            elif first_block_index == 8:
                start_idx = 1
            elif first_block_index == 15:
                start_idx = 0
            else:
                raise ValueError("Invalid arch_json start index")

        if start_idx == 3:
            x = features[3]
            blocks_to_use = self.blocks[0:]
        elif start_idx == 1:
            x = features[1]
            blocks_to_use = self.blocks[1:]
        elif start_idx == 0:
            x = features[0]
            blocks_to_use = self.blocks[2:]
        else:
            raise ValueError("Invalid start index chosen.")

        arch = []
        layer_counter = 1

        for i, block in enumerate(self.blocks):
            if block in blocks_to_use:
                block_arch = None
                if arch_json is not None:
                    block_arch = arch_json[(layer_counter - 1):(layer_counter - 1) + 7]
                x, block_arch_res = block(x, base_index=layer_counter, arch_spec=block_arch)
                arch.extend(block_arch_res)
            else:
                for i_null in range(7):
                    arch.append({
                        'index': layer_counter + i_null,
                        'type': None,
                        'op': None
                    })
            layer_counter += 7

        output = self.final_conv(x)
        return output, arch


if __name__ == "__main__":
    features = [
        torch.randn(1, 64, 128, 128),   # Feature 0
        torch.randn(1, 128, 64, 64),    # Feature 1
        torch.randn(1, 256, 32, 32),    # Feature 2
        torch.randn(1, 256, 32, 32),    # Feature 3
    ]


    for i in range(3):
        with open('5016ceb55bbc4a98853f4f7eb7d9c978.json', 'r') as f:
            loaded_arch = json.load(f)

        decoder = Decoder()
        output, arch_json = decoder(features, arch_json=loaded_arch)

        uid = uuid.uuid4().hex 
        filename = f"{i}_{uid}.json"
        filepath = os.path.join("stru_json", filename)

        with open(filepath, "w") as f:
            json.dump(arch_json, f, indent=2)