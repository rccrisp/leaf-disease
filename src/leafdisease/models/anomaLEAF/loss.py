from skimage.color import deltaE_ciede2000
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from functools import partial

import kornia

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class CIEDE2000_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img1: Tensor, img2: Tensor):

        # convert to L*A*B
        img1_lab = kornia.color.rgb_to_lab(img1)
        img2_lab = kornia.color.rgb_to_lab(img2)

        delE = deltaE_ciede2000(img1_lab.permute(0,2,3,1).cpu().detach().numpy(), img2_lab.permute(0,2,3,1).cpu().detach().numpy())

        return delE


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True):
        super().__init__()
        window = create_window(window_size, channel)
        self.ssim = partial(_ssim,
                            window=window.to(device),
                            window_size=window_size,
                            channel=channel,
                            size_average=size_average)

    def forward(self, Ii, Ir):
        ssim_loss = 1 - self.ssim(Ii, Ir)

        return ssim_loss

# Define Prewitt operator:
class Prewitt(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]) / 3
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).to(device)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x+1e-6)
        return x


# Define the gradient magnitude similarity map:
def GMS(Ii, Ir, edge_filter, median_filter, c=0.0026):
    x = torch.mean(Ii, dim=1, keepdim=True)
    y = torch.mean(Ir, dim=1, keepdim=True)
    g_I = edge_filter(median_filter(x))
    g_Ir = edge_filter(median_filter(y))
    g_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    return g_map


class MSGMS_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Prewitt(), median_filter=kornia.filters.MedianBlur((3, 3)))

    def GMS_loss(self, Ii, Ir):
        return torch.mean(1 - self.GMS(Ii, Ir))

    def forward(self, Ii, Ir):
        total_loss = self.GMS_loss(Ii, Ir)

        for _ in range(3):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            total_loss += self.GMS_loss(Ii, Ir)

        return total_loss / 4


class MSGMS_Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Prewitt(), median_filter=kornia.filters.MedianBlur((3, 3)))
        self.median_filter = kornia.filters.MedianBlur((21, 21))

    def GMS_Score(self, Ii, Ir):
        return self.GMS(Ii, Ir)

    def forward(self, Ii, Ir):
        total_scores = self.GMS_Score(Ii, Ir)
        img_size = Ii.size(-1)
        total_scores = F.interpolate(total_scores, size=img_size, mode='bilinear', align_corners=False)
        for _ in range(3):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            score = self.GMS_Score(Ii, Ir)
            total_scores += F.interpolate(score, size=img_size, mode='bilinear', align_corners=False)

        return (1 - total_scores) / 4