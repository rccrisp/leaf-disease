import torch
from typing import List
from torch import nn, Tensor

from leafdisease.criterions.msgms import MSGMSLoss
from leafdisease.components.unet import UNet
from leafdisease.utils.image import PatchMask, PatchedInputs, pad_nextpow2, mean_smoothing

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
        num_input_channels=3,
        num_disjoint_sets=3,
        k_values: List[int] = [2,4,8,16],
        reconstruct_k_idx: int = 0,
        blackout: bool = False,
    )-> None:
        super().__init__()

        assert reconstruct_k_idx < len(k_values)

        self.model = UNet(
            n_channels=num_input_channels
        )

        # function for generating masks
        self.k_list = k_values
        self.mask_gen = PatchMask(num_disjoint_sets, img_size = input_size, num_channels=3)
        self.input_gen = PatchedInputs(blackout=blackout)

        # for reconstruction
        self.reconstruction_k = k_values[reconstruct_k_idx]

        self.msgms_loss_func = MSGMSLoss()

    def forward(self, batch: Tensor):

         # pad image
        input = pad_nextpow2(batch["image"])
        input = input.to(device)

        # remove background
        foreground_mask = (input != 0).float()

        anomaly_map = 0
        # calculate anomaly score
        for k in self.k_list:
            # generate masks
            disjoint_masks = self.mask_gen(k)
            patched_inputs, inv_masks = self.input_gen(input, disjoint_masks)
        
            # model forward pass
            with torch.no_grad():
                outputs = [self.model(x) for x in patched_inputs]
            output = sum(map(lambda x, y: x * y * foreground_mask, outputs, inv_masks)) # recover all reconstructed patches

            if k == self.reconstruction_k:
                fake = output.clone()
        
            # score for this patch size
            anomaly_map += self.msgms_loss_func(input, output, as_loss=False)
        
        # smooth anomaly map
        anomaly_map = mean_smoothing(anomaly_map)

        # calculate the maximum heatmap score for each image in the batch
        max_values, _ = torch.max(anomaly_map, dim=1)

        return {"real": input, "fake": fake, "pred_heatmap": anomaly_map, "pred_label": max_values}



        
        