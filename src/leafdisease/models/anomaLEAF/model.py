import torch
from torch import nn, Tensor

from leafdisease.utils.image import pad_nextpow2, generate_masks

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
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

class Generator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
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
        out = nn.Sigmoid()(dec8)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, mean, std)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, input_tensor, target_tensor):
        x = torch.cat([input_tensor, target_tensor], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)

class anomaleafModel(nn.Module):
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
        batch_size: int,
        input_size: tuple[int, int],
        n_features: int,
        num_input_channels=3,
        k: int = 16
    )-> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_dim=num_input_channels,
            num_filter=n_features, 
            output_dim=num_input_channels
        )
        self.generator.normal_weight_init()

        self.discriminator: Discriminator = Discriminator(
            input_dim=num_input_channels*2,
            num_filter=n_features,
            output_dim=1
        )
        self.discriminator.normal_weight_init()

        self.mask_gen = generate_masks(k_list=[k], n=batch_size, im_size=input_size[0], num_channels=num_input_channels)

    def mask_input(self, batch: Tensor) -> Tensor:
        # create masks
        mask = next(self.mask_gen)
        assert mask[0].max() == 1, f"No input was masked ({mask[0].min()},{mask[0].max()})"

        # grayscale
        device = batch.device  # Get the device of the input batch
        grayscale = torch.mean(batch, dim=1, keepdim=True).to(device) # grayscale copy of batch
        grayscale = grayscale.repeat(1,3,1,1) # cast to three dimensions

        # created reduce image
        replace_mask = mask.eq(0)
        copy = batch.clone(device=batch.device)
        assert grayscale.size() == copy.size(), f"grayscale shape ({grayscale.size()}) does not match original shape ({copy.size()})"
        mask_A_batch = torch.where(replace_mask, grayscale, copy)
        assert mask_A_batch.size() == copy.size(), f"mask shape ({mask_A_batch.size()}) does not match original shape ({copy.size()})"

        replace_mask_inv = mask.eq(1)
        copy = batch.clone(device=batch.device)
        assert grayscale.size() == copy.size(), f"grayscale shape ({grayscale.size()}) does not match original shape ({copy.size()})"
        mask_B_batch = torch.where(replace_mask_inv, grayscale, copy)
        assert mask_B_batch.size() == copy.size(), f"mask shape ({mask_B_batch.size()}) does not match original shape ({copy.size()})"

        return mask_A_batch, mask_B_batch, mask

    def forward(self, batch: Tensor):
        """Get scores for batch
        
        Args:
            batch (Tensor): Images
        
        Returns: 
            Tensor: regeneration score
        """
        padded = pad_nextpow2(batch)

        # create masks
        mask_A, mask_B, mask = self.mask_input(padded)

        # regenerate from masks
        fake_A = self.generator(mask_A)
        fake_B = self.generator(mask_B)

        if self.training:
            return {"real": padded, "input_A": mask_A, "fake_A": fake_A, "input_B": mask_B, "fake_B": fake_B}
        else:
             # reconstruct image
            fake = fake_A.clone(device=batch.device)
            replace_mask = mask.eq(0)
            fake = torch.where(replace_mask, fake_A, fake_B)
            assert fake.size() == batch.size(), f"generated image ({fake.size()}) does not match original image ({batch.size()})"

            return {"real": padded.permute(0, 2, 3, 1).cpu().numpy(), "fake": fake.detach().permute(0, 2, 3, 1).cpu().numpy()}
      
