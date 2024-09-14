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
import torch.nn.functional as F

from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sos_token = args.sos_token
        self.vqgan = self.load_vqgan(args)
        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embed": 1024,
        }
        self.transformer = GPT(**transformer_config)
        self.p_keep = args.p_keep

    @classmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.vqgan_checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_imgs(self, indices, p1=16, p2=16):
        index_to_vectors = (
            self.vqgan.codebook.embedding(indices)
            .reshape(indices.shape[0], p1, p2, 256)
            .permute(0, 3, 1, 2)
        )
        image = self.vqgan.decode(index_to_vectors)
        return image.cpu().detach().numpy()[0].transpose(1, 2, 0)

    def forward(self, x):
        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long) * self.sos_token.to(
            x.device
        )

        mask = torch.bernoulli(
            self.p_keep * torch.ones(indices.shape, device=indices.device)
        )
        mask = mask.round().to(dtype=torch.int64)

        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = indices * mask + random_indices * (1 - mask)
        new_indices = torch.cat([sos_tokens, new_indices], dim=1)
        target = indices
        logits, _ = self.transformer(new_indices[:, :-1])
        return loss, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for _ in range(steps):
            logits, _ = self.transformer(x)
            # last time step
            logits = logits[:, -1, :] / temperature
            logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        x = x[:, c.shape[1] :]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()
        quant_z, indices = self.encode_to_z(x)
        sos_token = torch.ones(x.shape[0], 1) * self.sos_token
        sos_token = sos_token.long().to(x.device)

        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices, sos_token, steps=indices.shape[1] - start_indices.shape[1]
        )
        half_images = self.z_to_imgs(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_token, steps=indices.shape[1])
        full_images = self.z_to_imgs(sample_indices)

        x_rec = self.z_to_imgs(indices)

        log["x"] = x
        log["x_rec"] = x_rec
        log["half_images"] = half_images
        log["full_images"] = full_images

        return log, torch.cat((x, x_rec, half_images, full_images), dim=0)
