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

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm

from transformer import VQGANTransformer
from utils import load_data, plot_images


class TrainTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = VQGANTransformer(args).to(args.device)
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for m_name, module in self.model.transformer.named_modules():
            for p_name, param in module.named_parameters():
                full_name = f"{m_name}.{p_name}" if m_name else p_name
                if p_name.endswith("bias"):
                    no_decay.add(full_name)
                elif p_name.endswith("weight") and isinstance(
                    module, whitelist_weight_modules
                ):
                    decay.add(full_name)
                elif p_name.endswith("weight") and isinstance(
                    module, blacklist_weight_modules
                ):
                    no_decay.add(full_name)
        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=args.learning_rate, betas=(args.beta1, args.beta2)
        )
        return optimizer

    def train(self, args):
        train_dataset = load_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(args.device)
                    self.optimizer.zero_grad()
                    logits, target = self.model(imgs)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), target.view(logits.size(-1))
                    )
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix(
                        Transformer_loss=np.round(loss.cpu().detach().numpy().item(), 4)
                    )
                    pbar.update(0)

            log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(
                sampled_imgs,
                os.path.join("results", f"transformer_{epoch}.jpg"),
                nrow=4,
            )
            plot_images(log)
            torch.save(
                self.model.state_dict(),
                os.path.join("checkpoints", f"transformer_{epoch}.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--latent-dim", type=int, default=256, help="Latent dimension n_z."
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image height and width.)"
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.25, help="Commitment loss scalar."
    )
    parser.add_argument(
        "--image-channels", type=int, default=3, help="Number of channels of images."
    )
    parser.add_argument(
        "--dataset-path", type=str, default="./data", help="Path to data."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/last_ckpt.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device the training is on"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Input batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.25e-05, help="Learning rate."
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta param.")
    parser.add_argument("--beta2", type=float, default=0.9, help="Adam beta param.")
    parser.add_argument(
        "--disc-start", type=int, default=10000, help="When to start the discriminator."
    )
    parser.add_argument(
        "--disc-factor",
        type=float,
        default=1.0,
        help="Weighting factor for the Discriminator.",
    )
    parser.add_argument(
        "--l2-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for reconstruction loss.",
    )
    parser.add_argument(
        "--perceptual-loss-factor",
        type=float,
        default=1.0,
        help="Weighting factor for perceptual loss.",
    )

    parser.add_argument(
        "--pkeep",
        type=float,
        default=0.5,
        help="Percentage for how much latent codes to keep.",
    )
    parser.add_argument(
        "--sos-token", type=int, default=0, help="Start of Sentence token."
    )

    args = parser.parse_args()
    args.dataset_path = r"C:\Users\dome\datasets\flowers"
    args.checkpoint_path = r".\checkpoints\vqgan_last_ckpt.pt"

    train_transformer = TrainTransformer(args)
    train_transformer.train(args)
