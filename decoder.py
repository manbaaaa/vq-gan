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


class Decoder(nn.Module):
    def __init__():
        super().__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolution = [16]
        num_res_blocks = 3
        resolution = 16
        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]
        for i in range(len(channels)):
            out_channels = channels[i]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels, out_channels))
                resolution = resolution * 2

        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(
            nn.Conv2d(
                channels[-1], args.image_channels, kernel_size=3, stride=1, padding=1
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
