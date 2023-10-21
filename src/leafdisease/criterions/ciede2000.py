from skimage.color import deltaE_ciede2000
import kornia
from torch import Tensor
from torch.nn import Module

class CIEDE2000Loss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img1: Tensor, img2: Tensor):

        # convert to L*A*B
        img1_lab = kornia.color.rgb_to_lab(img1)
        img2_lab = kornia.color.rgb_to_lab(img2)

        delE = deltaE_ciede2000(img1_lab.permute(0,2,3,1).cpu().detach().numpy(), img2_lab.permute(0,2,3,1).cpu().detach().numpy())

        return delE/100