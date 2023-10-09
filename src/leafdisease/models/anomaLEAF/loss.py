import torch
from torch import nn, Tensor


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for pix2pix model"""

    def __init__(self) -> None:
           super().__init__()

           self.loss_bce = nn.BCELoss() 

    def forward(self, pred_real: Tensor, pred_fake: Tensor) -> Tensor:
        """Computer the loss for a predicted batch
        
        Args:
        pred_real (Tensor): Discriminator predictions for the real image.
        pred_fake (Tensor): Discriminator predictions for the fake image.

        Returns:
            Tensor: The computed discriminator loss.
           """
        error_discriminator_real = self.loss_bce(
            pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device)
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device)
        )
        loss_discriminator = (error_discriminator_fake + error_discriminator_real) * 0.5
        return loss_discriminator

class GeneratorLoss(nn.Module):
    """Generator loss for the pix2pix model.

    Args:
        wadv (int, optional): weight for adversarial loss. Defaults to 1
        wcon (int, optional): weight for pixel loss. Defaults to 100
    """

    def __init__(self, wadv=1, wcon=100) -> None:
        super().__init__()

        self.loss_patch = nn.BCELoss()
        self.loss_pixel = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon 

    def forward(
        self, patch_map: Tensor,  real: Tensor, fake: Tensor) -> Tensor:
        """Compute the loss for a batch.

        Args:
            patch_map (Tensor): feature map from discriminator
            real (Tensor): real image
            fake (Tensor): reconstructed image

        Returns:
            Tensor: The computed generator loss.
        """
        error_adv = self.loss_patch(patch_map,  torch.ones_like(patch_map))
        error_con = self.loss_pixel(real, fake)

        loss = error_adv*self.wadv + error_con*self.wcon
        
        return loss

