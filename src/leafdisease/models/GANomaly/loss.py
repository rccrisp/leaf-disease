"""Loss function for the GANomaly Model Implementation.

Code adapted from https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/models/ganomaly
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn


class GeneratorLoss(nn.Module):
    """Generator loss for the GANomaly model.

    Args:
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(self, wadv=1, wcon=50, wenc=1) -> None:
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self, latent_i: Tensor, latent_o: Tensor, images: Tensor, fake: Tensor, pred_real: Tensor, pred_fake: Tensor
    ) -> Tensor:
        """Compute the loss for a batch.

        Args:
            latent_i (Tensor): Latent features of the first encoder.
            latent_o (Tensor): Latent features of the second encoder.
            images (Tensor): Real image that served as input of the generator.
            fake (Tensor): Generated image.
            pred_real (Tensor): Discriminator predictions for the real image.
            pred_fake (Tensor): Discriminator predictions for the fake image.

        Returns:
            Tensor: The computed generator loss.
        """
        loss_enc = self.loss_enc(latent_i, latent_o)
        loss_con = self.loss_con(images, fake)
        loss_adv = self.loss_adv(pred_real, pred_fake)

        loss_generator = loss_adv * self.wadv + loss_con * self.wcon + loss_enc * self.wenc
        return loss_enc, loss_con, loss_adv, loss_generator


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for the GANomaly model."""

    def __init__(self) -> None:
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: Tensor, pred_fake: Tensor) -> Tensor:
        """Compute the loss for a predicted batch.

        Args:
            pred_real (Tensor): Discriminator predictions for the real image.
            pred_fake (Tensor): Discriminator predictions for the fake image.

        Returns:
            Tensor: The computed discriminator loss.
        """
        loss_real = self.loss_bce(
            pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device)
        )
        loss_fake = self.loss_bce(
            pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device)
        )
        loss_discriminator = (loss_fake + loss_real) * 0.5
        return loss_fake, loss_real, loss_discriminator