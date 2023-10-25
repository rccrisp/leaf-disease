import torch
from torch import Tensor
from torch.nn import Module

class ColourLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img1: Tensor, img2: Tensor):

        colour_loss = torch.abs(img1-img2)
        colour_loss = torch.sum(colour_loss, dim=1, keepdim=True)

        return colour_loss
