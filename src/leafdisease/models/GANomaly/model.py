"""Torch models defining encoder, decoder, Generator and Discriminator.

Code adapted from https://github.com/samet-akcay/ganomaly.
"""

# Copyright (c) 2018-2022 Samet Akcay, Durham University, UK
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import Tensor, nn

from leafdisease.utils.image import pad_nextpow2

class ganomalyModel(nn.Module):
    """GANomaly Model
    
    Args:
        batch_size (int): Batch size
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        k (int, optional): Tile size
    """
    def __init__(self,
        input_size: tuple[int, int],
        n_features: int,
        latent_vec_size: int,
        num_input_channels=3,
        extra_layers: bool = False,
        add_final_conv_layer: bool = True,
        threshold: float = float('-inf')
        )-> None:
        super().__init__()

        


        self.threshold = threshold

 

    def forward(self, batch):
        input = pad_nextpow2(batch)

        foreground_mask = ((input+1)/2 != 0).float()

        fake, latent_i, latent_o = self.generator(input)

        if self.training:
            fake = fake * foreground_mask - (1-foreground_mask)
            return {"real": input, "fake": fake, "latent_i": latent_i, "latent_o": latent_o}
        else:
            fake = fake * foreground_mask - (1-foreground_mask)
            score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
            label = self.threshold < score
            return {"real": input, "fake": fake, "pred_score": score, "pred_label": label}
