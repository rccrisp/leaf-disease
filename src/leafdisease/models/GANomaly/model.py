"""Torch models defining encoder, decoder, Generator and Discriminator.

Code adapted from https://github.com/samet-akcay/ganomaly.
"""

# Copyright (c) 2018-2022 Samet Akcay, Durham University, UK
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor, nn
from leafdisease.models.components.autoencoder import Encoder, Decoder

class Discriminator(nn.Module):
    """Discriminator.

        Made of only one encoder layer which takes x and x_hat to produce a score.

    Args:
        input_size (tuple[int, int]): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Add extra intermediate layers. Defaults to 0.
    """

    def __init__(
        self, input_size: tuple[int, int], num_input_channels: int, n_features: int, extra_layers: int = 0
    ) -> None:
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers, skips=False)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            elif isinstance(block, nn.ModuleList):
                for seq_block in block:
                    if isinstance(seq_block, nn.Sequential):
                        layers.extend(list(seq_block.children()))
                    else:
                        layers.append(seq_block)
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator model.

    Made of an encoder-decoder-encoder architecture.

    Args:
        input_size (tuple[int, int]): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
        unet (bool, optional): generator has unet architecture (skip layers across bottleneck)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        unet: bool = False
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer, skips=unet
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers, skips=unet)
        self.encoder2 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer, skips=False
        )

        self.unet = unet

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return generated image and the latent vectors."""

        if self.unet:
            latent_i, skips = self.encoder1(input_tensor)
            gen_image = self.decoder(latent_i, skips)
        else:
            latent_i = self.encoder1(input_tensor)
            gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o