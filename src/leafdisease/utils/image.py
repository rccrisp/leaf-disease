import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Tuple, List

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class PatchMask(nn.Module):

    def __init__(self,
        num_disjoint_sets,
        img_size: Tuple[int, int],
        num_channels=3
    )-> None:
        super().__init__()

        self.num_disjoint_sets = num_disjoint_sets
        self.img_h, self.img_w = img_size
        self.num_channels = num_channels

    def forward(self, patch_size):

        grid_h = math.ceil(self.img_h / patch_size)
        grid_w = math.ceil(self.img_w / patch_size)
        num_grids = grid_h * grid_w
        disjoint_masks = []

        for grid_idx in np.array_split(np.random.permutation(num_grids), self.num_disjoint_sets):
            flatten_mask = np.ones(num_grids)
            flatten_mask[grid_idx] = 0
            mask = flatten_mask.reshape((grid_h, grid_w))
            mask = mask.repeat(patch_size, axis=0).repeat(patch_size, axis=1)   # for all 3 channels
            mask = torch.tensor(mask, requires_grad=False, dtype=torch.float)
            mask = mask.to(device)
            disjoint_masks.append(mask)

        return disjoint_masks

class PatchedInputs(nn.Module):

    def __init__(self,
        blackout: bool
    )-> None:
        super().__init__()

        self.blackout = blackout

    def forward(self, input: Tensor, masks: List[Tensor]):
        inverse_masks = [1-mask for mask in masks]
        if self.blackout:
            patched_inputs = [input * mask for mask in masks]
        else:
            grayscale = torch.mean(input.clone(), dim=1, keepdim=True).to(device)
            grayscale = grayscale.expand(-1, 3, -1, -1)
            patched_inputs = [input * mask  + grayscale * inv_mask for mask, inv_mask in zip(masks, inverse_masks)]

        return patched_inputs, inverse_masks
        
def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    padded_batch = nn.functional.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch

def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)

