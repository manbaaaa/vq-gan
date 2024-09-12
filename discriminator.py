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

"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, num_layer=3):
        super().__init__()

        layers = [
            nn.Conv2d(
                args.image_channels,
                num_filters_last,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1
        for i in range(1, num_layer + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    kernel_size=4,
                    stride=2 if i < num_layer else 1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
            ]
            layers.append(
                nn.Conv2d(
                    num_filters_last * num_filters_mult,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
