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

import torch
import torch.nn as nn


class CodeBook(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_codebook_vector = args.num_codebook_vector
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vector, self.latent_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_codebook_vector, 1 / self.num_codebook_vector
        )

    def forward(self, z):
        # z: B, C, H, W
        z = z.permute(0, 2, 3, 1).contiguous()
        # z: B * H * W, C
        z_flattened = z.view(-1, self.latent_dim)
        # l2 distance, (a-b)^2 = a^2 + b^2 - 2ab
        # (B * H * W, 1) + (B * H * W,) - 2 * (B * H * W, num_codebook_vector)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # https://spaces.ac.cn/archives/9826
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss
