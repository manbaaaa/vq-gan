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

from codebook import Codebook
from decoder import Decoder
from encoder import Encoder


class VQGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args).to(args.device)
        self.codebook = Codebook(args).to(args.device)
        self.decoder = Decoder(args).to(args.device)
        self.quan_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)
        self.post_quan_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)

    def forward(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quan_conv_encoded_imgs = self.quan_conv(encoded_imgs)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quan_conv_encoded_imgs
        )
        post_quan_conv_codebook_mapping = self.post_quan_conv(codebook_mapping)
        decoded_imgs = self.decoder(post_quan_conv_codebook_mapping)
        return decoded_imgs, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quan_conv_encoded_imgs = self.quan_conv(encoded_imgs)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quan_conv_encoded_imgs
        )
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quan_conv_codebook_mapping = self.post_quan_conv(z)
        decoded_imgs = self.decoder(post_quan_conv_codebook_mapping)
        return decoded_imgs

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.layers[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]
        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return λ * 0.8

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.0):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        return self.load_state_dict(torch.load(path))
