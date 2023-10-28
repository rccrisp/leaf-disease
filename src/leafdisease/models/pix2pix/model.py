import torch
from torch import nn, Tensor

from leafdisease.utils.image import pad_nextpow2, denormalise
from leafdisease.components.pix2pix import Generator
from leafdisease.criterions.colour import ColourLoss

class pix2pixModel(nn.Module):
    """AnomaLEAF Model
    
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
        n_features: int,
        num_input_channels=1,
        num_output_channels=3,
    )-> None:
        super().__init__()

        self.generator: Generator = Generator(
            input_dim=num_input_channels,
            num_filter=n_features, 
            output_dim=num_output_channels
        )

        self.colour_loss_func = ColourLoss()

    
    def forward(self, batch: Tensor):

        target = pad_nextpow2(batch["image"])
        
        input = torch.mean(target, dim=1, keepdim=True)

        output = self.generator(input)
        
        colour_map = self.colour_loss_func(target, output)

        colour_score, _ = torch.max(colour_map.view(colour_map.size(0), -1), dim=1)

        output = denormalise(output)
        target = denormalise(target)

        return {"real": target, "fake": output, "anomaly_map": colour_map, "anomaly_score": colour_score}


      
