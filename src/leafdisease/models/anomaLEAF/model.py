import torch
from typing import List
from torch import nn, Tensor
from scipy.ndimage import gaussian_filter
import random

from leafdisease.utils.image import pad_nextpow2, generate_masks

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.relu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_size)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out

class UNet(nn.Module):
    def __init__(self, input_dim, num_filter=64, output_dim=3):
        super(UNet, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv8 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)

        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)

        # Final layer
        out = nn.Tanh()(dec8)

        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, mean, std)

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
        n_features: int,
        anomaly_score_func,
        num_input_channels=3,
        k_values: List[int] = [2,4,8,16],
        blackout: bool = False,
    )-> None:
        super().__init__()
        self.generator: UNet = UNet(
            input_dim=num_input_channels,
            num_filter=n_features, 
            output_dim=num_input_channels
        )
        self.generator.normal_weight_init()

        self.anomaly_score = anomaly_score_func

        # Mask information
        self.k_values = k_values
        self.image_size = input_size[0]
        self.blackout = blackout

    def generate_inputs(self, original, masks, blackout=False):

        if blackout:
            inputs =  [original * mask.clone().detach().requires_grad_(False) for mask in masks]
        # image inpainting
        else :
            grayscale = torch.mean((original.clone()), dim=1, keepdim=True)
            grayscale = grayscale.expand(-1,3,-1,-1)
            inputs =  [original * mask.clone().detach().requires_grad_(False) + grayscale * (1-mask.clone().detach().requires_grad_(False)) for mask in masks]
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

            outputs = [self.generator(x) for x in inputs]
            output = sum(map(lambda x, y: x * (1 - y.clone().detach().requires_grad_(False))*leaf_segment, outputs, masks))
            # output = output * leaf_segment

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

                inputs = [padded * mask.clone().detach().requires_grad_(False) for mask in masks]
                outputs = [self.generator(x) for x in inputs]
                output = sum(map(lambda x, y: x * (1 - y.clone().detach().requires_grad_(False))*leaf_segment, outputs, masks))
                # output = output * leaf_segment

                score += self.anomaly_score(padded, output) / (N**2)

            score = score.detach().squeeze().cpu().numpy()
            for i in range(score.shape[0]):
                score[i] = gaussian_filter(score[i], sigma=7)
            real = padded
            fake = output
            # fake = output.cpu.numpy()
            # real = output.cpu.numpy()

            return {"real": real, "fake": fake, "pred_score": score.max(), "pred_heatmap": score}


        
        