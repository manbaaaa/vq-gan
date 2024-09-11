#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn

from helper import (
    DownSampleBlock,
    GroupNorm,
    NonLocalBlock,
    ResidualBlock,
    Swish,
    UpSampleBlock,
)


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolution = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [
            nn.Conv2d(
                args.image_channels, channels[0], kernel_size=3, stride=1, padding=1
            )
        ]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(in_channels, out_channels))
                resolution = resolution // 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(
            nn.Conv2d(
                channels[-1], args.latent_channels, kernel_size=3, stride=1, padding=1
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
