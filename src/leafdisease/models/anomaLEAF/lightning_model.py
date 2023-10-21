"""anomaLEAF: an unsupervised anomaly detection model based on colour reconstruction

"""

from __future__ import annotations

import os
import logging
from typing import List
import random

import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch import Tensor, optim
import torchvision

from leafdisease.components.unet import UNet
from leafdisease.criterions.msgms import MSGMSLoss
from leafdisease.criterions.ssim import SSIMLoss
from leafdisease.utils.image import PatchMask, PatchedInputs, pad_nextpow2, mean_smoothing

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
        blackout: bool = False,
        k_values: List[int] = [2,4,8,16],
        num_input_channels=3,
        num_disjoint_sets=3,
        gamma: int = 1,
        alpha: int = 1,
        tau: int = 1,
        epsilon: int = 1,
        lr: float = 0.0001,
        beta1: float = 0.5,
        beta2: float = 0.999,
        save_examples_every_n_epochs: int = 10,
        example_images: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.model = UNet(
            n_channels=num_input_channels
        )

        # function for generating masks
        self.k_list = k_values
        self.mask_gen = PatchMask(num_disjoint_sets, img_size = input_size, num_channels=3)
        self.input_gen = PatchedInputs(blackout=blackout)

        # Loss functions
        # self.l1_loss_func = nn.L1Loss(reduction="mean")
        self.l2_loss_func = nn.MSELoss(reduction="mean")
        self.ssim_loss_func = SSIMLoss()
        self.msgms_loss_func = MSGMSLoss()

        # Loss parameters
        self.epsilon = epsilon
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
            self.model.parameters(),
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

        # pad image
        input = pad_nextpow2(batch["image"])

        # remove background
        foreground_mask = ((input+1)/2 != 0).float()

        # generate masks
        k = random.sample(self.k_list, 1)
        disjoint_masks = self.mask_gen(k[0])
        patched_inputs, inv_masks = self.input_gen(input, disjoint_masks)
        
        # model forward pass
        outputs = [self.model(x) for x in patched_inputs]
        output = sum(map(lambda x, y: x * y * foreground_mask, outputs, inv_masks)) # recover all reconstructed patches

        # loss
        l2_loss = self.l2_loss_func(input, output)
        gms_loss = self.msgms_loss_func(input, output)
        ssim_loss = self.ssim_loss_func(input, output)

        loss = self.gamma * l2_loss + self.alpha * gms_loss + self.tau * ssim_loss

        # Log
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_l2_loss", l2_loss, on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gms_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_ssim_loss", ssim_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        
        return loss

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """

        # pad image
        input = pad_nextpow2(batch["image"])

        # remove background
        foreground_mask = ((input+1)/2 != 0).float()

        # generate masks
        disjoint_masks = self.mask_gen(self.k_list[0])
        patched_inputs, inv_masks = self.input_gen(input, disjoint_masks)
        
        # model forward pass
        outputs = [self.model(x) for x in patched_inputs]
        output = sum(map(lambda x, y: x * y * foreground_mask, outputs, inv_masks)) # recover all reconstructed patches

        # loss
        l2_loss = self.l2_loss_func(input, output)
        gms_loss = self.msgms_loss_func(input, output)
        ssim_loss = self.ssim_loss_func(input, output)

        loss = self.gamma * l2_loss + self.alpha * gms_loss + self.tau * ssim_loss

        anomaly_map = 0
        # calculate anomaly score
        for k in self.k_list:
            # generate masks
            disjoint_masks = self.mask_gen(k)
            patched_inputs, inv_masks = self.input_gen(input, disjoint_masks)
        
            # model forward pass
            outputs = [self.model(x) for x in patched_inputs]
            output = sum(map(lambda x, y: x * y * foreground_mask, outputs, inv_masks)) # recover all reconstructed patches
        
            # score for this patch size
            anomaly_map += self.msgms_loss_func(input, output, as_loss=False)
        
        # smooth anomaly map
        anomaly_map = mean_smoothing(anomaly_map)

        # calculate the maximum heatmap score for each image in the batch
        max_values, _ = torch.max(anomaly_map, dim=1)

        # Calculate the mean of the maximum values
        mean_max_value = torch.mean(max_values)
        
        # Log
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_l2_loss", l2_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gms_loss", gms_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_ssim_loss", ssim_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_anomaly_score", mean_max_value.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        
        return loss

    def generate_and_save_samples(self, epoch):
        # Generate and save example images

        with torch.no_grad():

            # pad image
            input = pad_nextpow2(self.example_images)

            # remove background
            foreground_mask = ((input+1)/2 != 0).float()

            # generate masks
            disjoint_masks = self.mask_gen(self.k_list[0])
            patched_inputs, inv_masks = self.input_gen(input, disjoint_masks)
        
            # model forward pass
            outputs = [self.model(x) for x in patched_inputs]
            output = sum(map(lambda x, y: x * y * foreground_mask, outputs, inv_masks))    # recover all reconstructed patches

            # Convert the output to [0, 1] range for the entire batch
            output = (output + 1) / 2

            num_samples = self.example_images.size(0)

            grid = torchvision.utils.make_grid(output, nrow=int(num_samples**0.5))
            filename = f"anomaLEAF_fake_epoch={epoch}.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)
            
    def on_validation_epoch_end(self):

        # Specify when to generate and save examples (e.g., every n epochs)
        if self.example_images is not None:

            if self.current_epoch % self.save_n_epochs == 0:

                # Call the method to generate and save examples
                self.generate_and_save_samples(self.current_epoch)