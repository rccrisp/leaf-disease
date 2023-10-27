import torch
from typing import List
from torch import nn, Tensor

from leafdisease.criterions.msgms import MSGMSLoss
from leafdisease.criterions.colour import ColourLoss
from leafdisease.components.unet import UNet
from leafdisease.utils.image import PatchMask, PatchedInputs, pad_nextpow2, mean_smoothing

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
        num_input_channels=3,
        num_disjoint_sets=3,
        k_values: List[int] = [2,4,8,16],
        blackout: bool = False,
    )-> None:
        super().__init__()

        self.model = UNet(
            n_channels=num_input_channels
        )

        # function for generating masks
        self.k_list = k_values
        self.mask_gen = PatchMask(num_disjoint_sets, img_size = input_size, num_channels=3)
        self.input_gen = PatchedInputs(blackout=blackout)

        self.msgms_loss_func = MSGMSLoss()
        self.colour_loss_func = ColourLoss()

    def forward(self, batch: Tensor):

         # pad image
        image = pad_nextpow2(batch["image"])

        foreground_mask = pad_nextpow2(batch["mask"])
        
        input = image*foreground_mask - (1-foreground_mask)

        anomaly_map = 0
        colour_map = 0
        fake = {}
        # calculate anomaly score
        for k in self.k_list:
            # generate masks
            disjoint_masks = self.mask_gen(k)
            patched_inputs, inv_masks = self.input_gen((input+1)/2, disjoint_masks)
        
            # model forward pass
            with torch.no_grad():
                outputs = [self.model((x-1)/2) for x in patched_inputs]
            output = sum(map(lambda x, y: x * y, outputs, inv_masks)) # recover all reconstructed patches
            output = output * foreground_mask - (1-foreground_mask)

            fake[k] = output
        
            # anomaly score for this patch size
            anomaly_map += self.msgms_loss_func(input, output, as_loss=False)

            colour_map += self.colour_loss_func(input, output)
        
        # smooth anomaly map
        anomaly_map = mean_smoothing(anomaly_map)

        # calculate the maximum heatmap score for each image in the batch
        anomaly_score, _ = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)

        # smooth colour map
        colour_map = mean_smoothing(colour_map)
        colour_map /= len(self.k_list)

        colour_score, _ = torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)

        return {"real": input, "fake": fake, "anomaly_map": anomaly_map, "anomaly_score": anomaly_score, "colour_map": colour_map, "color_score": colour_score}



        
        