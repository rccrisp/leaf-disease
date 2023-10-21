import torch
from typing import List
from torch import nn, Tensor
from scipy.ndimage import gaussian_filter
import random
from leafdisease.components.unet import UNet

from leafdisease.utils.image import pad_nextpow2, generate_masks

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class anomaleafModel(nn.Module):
    """AnomaLEAF Model
    
    Args:
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features extracted at each CNN layer.
        anomaly_score_func (func): calculates anomaly score.
        num_input_channels (int): number of input features. For images it is equal to the number of channels
        k_values (int, optional): Tile size.
        blackout (bool, optional): specifies the type of inpainting (colour or blackout)
    """
    def __init__(self,
        input_size: tuple[int, int],
        anomaly_score_func,
        num_input_channels=3,
        k_values: List[int] = [2,4,8,16],
        blackout: bool = False,
    )-> None:
        super().__init__()
        self.unet: UNet = UNet(
            n_channels=num_input_channels
        )

        self.anomaly_score = anomaly_score_func

        # Mask information
        self.k_values = k_values
        self.image_size = input_size[0]
        self.blackout = blackout

    def generate_inputs(self, original, masks, blackout=False):
        if blackout:
            inputs = [original * mask.detach().requires_grad_(False) for mask in masks]
        else:
            grayscale = torch.mean(original.clone(), dim=1, keepdim=True)
            grayscale = grayscale.expand(-1, 3, -1, -1)
            inputs = [original * mask.detach().requires_grad_(False) + grayscale * (1 - mask.detach().requires_grad_(False)) for mask in masks]
        return inputs

    def forward(self, batch: Tensor):
        """Get scores for batch
        
        Args:
            batch (Tensor): Images
        
        Returns: 
            Tensor: regeneration score
        """
        padded = pad_nextpow2(batch)

        leaf_segment = (padded != 0).float()

        if self.training:
            k_list = random.sample(self.k_values, 1)
            mask_generator = generate_masks(k_list=k_list,n= 3, im_size = self.image_size)
            masks = next(mask_generator)
            masks = masks.to(device)

            inputs = self.generate_inputs(padded, masks, self.blackout)

            outputs = [self.unet(x) for x in inputs]
            output = sum(map(lambda x, y: x * (1 - y.clone().detach().requires_grad_(False))*leaf_segment, outputs, masks))

            return {"real": padded, "fake": output}
        else :
            score = 0
            for k in self.k_values:
                mask_generator = generate_masks(k_list=[k],n= 3, im_size = self.image_size)
                masks = next(mask_generator)
                masks = masks.to(device)

                inputs = self.generate_inputs(padded, masks, self.blackout)
                
                img_size = padded.size(-1)
                N = img_size // k

                outputs = [self.unet(x) for x in inputs]
                output = sum(map(lambda x, y: x * (1 - y.clone().detach().requires_grad_(False))*leaf_segment, outputs, masks))

                score += self.anomaly_score(padded, output) / (N**2)

            score = score.detach().squeeze().cpu().numpy()
            for i in range(score.shape[0]):
                score[i] = gaussian_filter(score[i], sigma=7)
            real = padded
            fake = output

            return {"real": real, "fake": fake, "pred_heatmap": score}


        
        