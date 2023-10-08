import math 
from typing import List, Tuple

import torch
from torch import Tensor, nn

class Encoder(nn.Module):
    """Encoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
        skips (bool): Use skip layers to ferry features across the bottleneck as in Unet. Defaults to False.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        skips: bool = False
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = []
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            pyramid_step = nn.Sequential()
            pyramid_step.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            pyramid_step.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            pyramid_step.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            self.pyramid_features.append(pyramid_step)
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Convert the list to a ModuleList for proper PyTorch tracking
        self.pyramid_features = nn.ModuleList(self.pyramid_features)

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

        self.skips = skips

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, List[Tensor]] | Tensor:
        """Return latent vectors."""

        # input layer
        output = self.input_layers(input_tensor)

        # extra layers
        output = self.extra_layers(output)

        # downsampling and skips
        skips = []
        for down_layer in self.pyramid_features:
            output = down_layer(output)
            skips.append(output)
        # reverse skips
        skips = reversed(skips[:-1])

        # final layer
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        if self.skips:
            return output, skips
        else: 
            return output


class Decoder(nn.Module):
    """Decoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
        skips (bool): Use skip layers to ferry features across the bottleneck as in Unet. Defaults to False.
        dropout (int): Number of dropout layers, used to introduce noise to GAN training. Defaults to 0.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        skips: bool = False,
        dropout: int = 0
    ) -> None:
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = []
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            inverse_pyramid_step = nn.Sequential()
            inverse_pyramid_step.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            inverse_pyramid_step.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            inverse_pyramid_step.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            if dropout > 0:
                inverse_pyramid_step.add_module((f'pyramid-{out_features}-dropout', nn.Dropout(p=0.5)))
                dropout -= 1
            self.inverse_pyramid.append(inverse_pyramid_step)
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Convert the list to a ModuleList for proper PyTorch tracking
        self.inverse_pyramid = nn.ModuleList(self.inverse_pyramid)

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features)
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu", nn.LeakyReLU(0.2, inplace=True)
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-sigmoid", nn.Sigmoid())

        # for Unet architecture
        self.skips = skips

    def forward(self, input, skips=None) -> Tensor:
        """Return generated image."""

        # receive skips if we expect skips
        assert (self.skips == (skips is not None))

        # input layer
        output = self.latent_input(input)

        # upsampling and skips
        if self.skips:
            for up_layer, skip in zip(self.inverse_pyramid, skips):
                output = up_layer(output)
                output = torch.cat((output,skip), dim=1)
        else:
            for up_layer in self.inverse_pyramid:
                output = up_layer(output)

        # extra layers
        output = self.extra_layers(output)

        # final layer
        output = self.final_layers(output)

        return output
