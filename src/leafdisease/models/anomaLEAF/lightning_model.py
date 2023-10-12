"""anomaLEAF: an unsupervised anomaly detection model based on colour reconstruction

"""

from __future__ import annotations

import os
import logging
import matplotlib.pyplot as plt
from typing import List

import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch import Tensor, optim
import torchvision

from .loss import SSIM_Loss, MSGMS_Loss, MSGMS_Score
from .model import anomaleafModel

logger = logging.getLogger(__name__)

class anomaLEAF(pl.LightningModule):
    """PL Lightning Module for the anomaLEAF Algorithm.

    Args:
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 100.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        n_features: int,
        blackout: bool = False,
        k_values: List[int] = [2,4,8,16],
        num_input_channels=3,
        gamma: int = 1,
        alpha: int = 1,
        tau: int = 1,
        lr: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        save_examples_every_n_epochs: int = 10,
        example_images: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.model: anomaleafModel = anomaleafModel(
            input_size=input_size,
            n_features=n_features,
            k_values=k_values,
            anomaly_score_func=MSGMS_Score(),
            num_input_channels=num_input_channels,
            blackout=blackout
        )

        # Loss functions
        self.l2_loss_func = nn.MSELoss(reduction="mean")
        self.ssim_loss_func = SSIM_Loss()
        self.msgms_loss_func = MSGMS_Loss()

        # Loss parameters
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau


        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # for visualising GAN training
        self.save_n_epochs = save_examples_every_n_epochs
        self.example_images = example_images
        self.save_example_dir = save_example_dir

    def configure_optimizers(self) -> optim.Optimizer:
        """Configures optimizers autoencoder.

        Returns:
            Optimizer: Adam optimizer for autoencoder
        """
        optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        return optimizer
    

    def training_step(
        self, batch: dict[str, str | Tensor], batch_idx: int
    ) :
        """Training step.

        Args:
            batch (dict[str, str | Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            
        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.

        # forward step
        output = self.model(batch["image"])
        real = output["real"]
        fake = output["fake"]

        # loss
        l2_loss = self.l2_loss_func(real, fake)
        gms_loss = self.msgms_loss_func(real, fake)
        ssim_loss = self.ssim_loss_func(real, fake)

        loss = self.gamma * l2_loss + self.alpha * gms_loss + self.tau * ssim_loss

        # Log
        self.log("train_l2_loss", l2_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gms_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_ssim_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        
        return loss

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """

         # forward step
        output = self.model(batch["image"])
        real = output["real"]
        fake = output["fake"]

        # loss
        l2_loss = self.l2_loss_func(real, fake)
        gms_loss = self.msgms_loss_func(real, fake)
        ssim_loss = self.ssim_loss_func(real, fake)

        loss = self.gamma * l2_loss + self.alpha * gms_loss + self.tau * ssim_loss

        # Log
        self.log("val_l2_loss", l2_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gms_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_ssim_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        
        return loss


    def generate_and_save_samples(self, epoch):
        # Generate and save example images
        self.eval()  # Set the model to evaluation mode to ensure deterministic results
        with torch.no_grad():
            # Generate samples from your GAN
            generated_samples = self.model(self.example_images)

            # Convert generated samples to a grid for visualization (using torchvision)
            num_samples = self.example_images.size(0)
            fake = generated_samples["fake"]
            grid = torchvision.utils.make_grid(fake, nrow=int(num_samples**0.5))
            filename = f"anomaLEAF_fake_epoch={epoch}.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)
            
        self.train()  # Set the model back to training mode 
    
    def on_validation_epoch_end(self):

        # Specify when to generate and save examples (e.g., every n epochs)
        if self.example_images is not None:

            if self.current_epoch % self.save_n_epochs == 0:

                # Call the method to generate and save examples
                self.generate_and_save_samples(self.current_epoch)