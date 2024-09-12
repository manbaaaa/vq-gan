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

from discriminator import Discriminator
from lpips import LPIPS
from utils import weights_init
from vqgan import VQGAN


class TrainVQGAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vqgan = VQGAN(args).to(args.device)
        self.discriminator = Discriminator(args).to(args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.lr
        opt_vq = torch.optim.AdamW(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.quan_conv.parameters())
            + list(self.vqgan.post_quan_conv.parameters()),
            lr=lr,
            eps=1e-8,
            betas=(args.beta1, args.beta2),
        )
        opt_disc = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=args.lr,
            eps=1e-8,
            betas=(args.beta1, args.beta2),
        )
        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def train(self, args):
        train_dataset = load_data(args)
        step_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(args.device)
                    decoded_imgs, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_imgs)

                    disc_factor = self.vqgan.adopt_weight(
                        args.disc_factor, epoch * step_per_epoch + i, args.disc_start
                    )

                    perceptual_loss = self.perceptual_loss(decoded_imgs, imgs)
                    rec_loss = troch.abs(imgs - decoded_imgs)
                    perceptual_rec_loss = (
                        args.perceptual_loss_factor * perceptual_loss
                        + args.rec_loss_factor * rec_loss
                    )
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    # generator loss
                    # the generator's objective is to produce images that can deceive the discriminator.
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = q_loss + perceptual_rec_loss + λ * disc_factor * g_loss

                    # hinge loss L(y, f(x)) = max(0, 1 - y * f(x))
                    d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                    d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                    gan_loss = (d_loss_real + d_loss_fake) * 0.5 * disc_factor

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat(
                                (imgs[:4], decoded_images.add(1).mul(0.5)[:4])
                            )
                            vutils.save_image(
                                real_fake_images,
                                os.path.join("results", f"{epoch}_{i}.jpg"),
                                nrow=4,
                            )

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                    )
                    pbar.update(0)
            torch.save(
                self.vqgan.state_dict(),
                os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension n_z (default: 256)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image height and width (default: 256)",
    )
    parser.add_argument(
        "--num-codebook-vectors",
        type=int,
        default=1024,
        help="Number of codebook vectors (default: 256)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Commitment loss scalar (default: 0.25)",
    )
    parser.add_argument(
        "--image-channels",
        type=int,
        default=3,
        help="Number of channels of images (default: 3)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data",
        help="Path to data (default: /data)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device the training is on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Input batch size for training (default: 6)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.25e-05,
        help="Learning rate (default: 0.0002)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Adam beta param (default: 0.0)"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.9, help="Adam beta param (default: 0.999)"
    )
    parser.add_argument(
        "--disc-start",
        type=int,
        default=10000,
        help="When to start the discriminator (default: 0)",
    )
    parser.add_argument("--disc-factor", type=float, default=1.0, help="")
    parser.add_argument(
        "--rec-loss-factor",
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

    args = parser.parse_args()
    args.dataset_path = r"C:\Users\dome\datasets\flowers"

    train_vqgan = TrainVQGAN(args)
