import math 
from typing import List, Tuple

import torch
from torch import Tensor, nn

class Encoder(nn.Module):
    """Encoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_features (int): number of features of latent vector z
        latent_vec_dim (int): dimensions of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
        skips (bool): Use skip layers to ferry features across the bottleneck as in Unet. Defaults to False.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_features: int,
        latent_vec_dim: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        skips: bool = False
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.ConvTranspose2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=0, bias=False),
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
        while pyramid_dim > latent_vec_dim:
            in_features = n_features
            out_features = n_features * 2
            pyramid_step = nn.Sequential()
            pyramid_step.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=0, bias=False),
            )
            pyramid_step.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            pyramid_step.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            self.pyramid_features.append(pyramid_step)
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Convert the list to a ModuleList for proper PyTorch tracking
        self.pyramid_features = nn.ModuleList(self.pyramid_features)

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
        latent_vec_features: int,
        latent_vec_dim: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        skips: bool = False,
        dropout: int = 0
    ) -> None:
        super().__init__()

        # for Unet architecture
        self.skips = skips

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // latent_vec_dim, 2)) - 1
        n_input_features = n_features * (2**exp_factor)

        # Create inverse pyramid
        self.inverse_pyramid = []
        scale_factor = 1
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > latent_vec_dim:
            in_features = n_input_features * scale_factor
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

            scale_factor = 2 if self.skips else 1

        # Convert the list to a ModuleList for proper PyTorch tracking
        self.inverse_pyramid = nn.ModuleList(self.inverse_pyramid)

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.ConvTranspose2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=0, bias=False),
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
                padding=0,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-sigmoid", nn.Sigmoid())

    def forward(self, input, skips=None) -> Tensor:
        """Return generated image."""

        # receive skips if we expect skips
        assert (self.skips == (skips is not None))

        output = input

        # upsampling and skips
        if self.skips:
            for up_layer, skip in zip(self.inverse_pyramid, skips):
                output = up_layer(output)   
                assert output.shape == skip.shape, f"layer ({output.shape}) does not match skip ({skip.shape})"
                output = torch.cat((output,skip), dim=1)
                print(f"{output.shape} {skip.shape}")
        else:
            for up_layer in self.inverse_pyramid:
                output = up_layer(output)

        # extra layers
        output = self.extra_layers(output)

        # final layer
        output = self.final_layers(output)

        return output
