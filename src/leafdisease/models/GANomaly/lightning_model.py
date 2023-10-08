"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch import Tensor, optim

from leafdisease.utils.image import pad_nextpow2
from .model import Generator, Discriminator
from .loss import GeneratorLoss, DiscriminatorLoss

logger = logging.getLogger(__name__)

class Ganomaly(pl.LightningModule):
    """PL Lightning Module for the GANomaly Algorithm.

    Args:
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        n_features: int,
        latent_vec_size: int,
        num_input_channels=3,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        visualise_training=False
    ) -> None:
        super().__init__()

        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.weights_init(self.generator)
        
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        self.weights_init(self.discriminator)

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.visualise_training = visualise_training
        self.example_image = None

        # important for training with multiple optimizers
        self.automatic_optimization = False

    @staticmethod
    def weights_init(module: torch.nn.Module) -> None:
        """Initialize DCGAN weights.

        Args:
            module (nn.Module): [description]
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(module.bias.data, 0)


    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return [optimizer_d, optimizer_g]
    
    def forward(self, batch):
        padded_batch = pad_nextpow2(batch)

        fake, latent_i, latent_o = self.generator(padded_batch)

        return padded_batch, fake, latent_i, latent_o
        
        

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

        disc_optimiser, gen_optimiser = self.optimizers()

        ##########################
        # Optimize Discriminator #
        ##########################
        padded, fake, _, _ = self(batch["image"])

        pred_real, _ = self.discriminator(padded)
        pred_fake, _ = self.discriminator(fake)

        # loss
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        # Discriminator grad calculations
        disc_optimiser.zero_grad()
        self.manual_backward(disc_loss)
        disc_optimiser.step()

        ######################
        # Optimize Generator #
        ######################
        padded, fake, latent_i, latent_o = self(batch["image"])

        _, feature_real = self.discriminator(padded)
        _, feature_fake = self.discriminator(fake)
        
        # loss
        gen_loss = self.generator_loss(latent_i, latent_o, padded, fake, feature_real, feature_fake)

        # Generator grad calculations
        gen_optimiser.zero_grad()
        self.manual_backward(gen_loss)
        gen_optimiser.step()

        # Log
        self.log("train_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        
        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        
        if self.visualise_training and self.example_image == None:
            self.example_image = batch["image"][0]
            self.example_image = self.example_image.unsqueeze(0)

        padded, fake, latent_i, latent_o = self(batch["image"])

        # calculate the anomaly score
        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        batch_score = torch.mean(score)

        pred_real, _ = self.discriminator(padded)

        pred_fake, _ = self.discriminator(fake.detach())
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        pred_fake, _ = self.discriminator(fake)
        gen_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        # log
        self.log("val_score", batch_score.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
    
    def predict_step(self, batch: dict[str, str | Tensor]):

        # validation step is used for inference
        padded, fake, latent_i, latent_o = self(batch["image"])

        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)

        return {"real": padded, "generated": fake, "anomaly_score": score, "filename": batch["filename"]}

    def reconstruct_and_plot(self):
        # Pass the validation image through the GAN for reconstruction
        padded, fake, _, _ = self(self.example_image)

        # Plot the original and reconstructed images in color
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(padded[0].permute(1, 2, 0).detach().cpu())
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(fake[0].permute(1, 2, 0).detach().cpu())
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def on_validation_epoch_end(self):
        if self.visualise_training:
            self.reconstruct_and_plot()
    
