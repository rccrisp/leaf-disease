"""GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

https://arxiv.org/abs/1805.06725
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
import pytorch_lightning as pl
from torch import Tensor, optim
import torchvision
import os

from leafdisease.utils.image import pad_nextpow2
from leafdisease.components.GAN import Generator, Discriminator
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
        depth: int = 4,
        num_input_channels=3,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        save_examples_every_n_epochs: int = 10,
        example_images: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            depth=depth,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )

        self.weights_init(self.generator)

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)

        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        
        self.weights_init(self.discriminator)

        self.discriminator_loss = DiscriminatorLoss()

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # for visualising GAN training
        self.save_n_epochs = save_examples_every_n_epochs
        self.example_images = example_images
        self.save_example_dir = save_example_dir

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
        
        input = pad_nextpow2(batch["image"])

        ##########################
        # Optimize Discriminator #
        ##########################
        self.toggle_optimizer(disc_optimiser)
        with torch.no_grad():
            fake, latent_i, latent_o = self.generator(input)

        pred_real, _ = self.discriminator(input)
        pred_fake, _ = self.discriminator(fake)

        # loss
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        # Discriminator grad calculations
        self.manual_backward(disc_loss)
        disc_optimiser.step()
        disc_optimiser.zero_grad()
        self.untoggle_optimizer(disc_optimiser)

        ######################
        # Optimize Generator #
        ######################
        self.toggle_optimizer(gen_optimiser)
        fake, latent_i, latent_o = self.generator(input)

        with torch.no_grad():
            _, feature_real = self.discriminator(input)
            _, feature_fake = self.discriminator(fake)

        # loss
        gen_loss = self.generator_loss(latent_i, latent_o, input, fake, feature_real, feature_fake)

        # Generator grad calculations
        self.manual_backward(gen_loss)
        gen_optimiser.step()
        gen_optimiser.zero_grad()
        self.untoggle_optimizer(gen_optimiser)

        #################
        # Anomaly Score #
        #################
        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        batch_score, _ = torch.max(score, dim=0)

        # Log
        self.log("train_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_score", batch_score.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """

        input = pad_nextpow2(batch["image"])

        fake, latent_i, latent_o = self.generator(input)

        ######################
        # Discriminator Loss #
        ######################
        pred_real, _ = self.discriminator(input)
        pred_fake, _ = self.discriminator(fake)
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        ##################
        # Generator Loss #
        ##################
        gen_loss = self.generator_loss(latent_i, latent_o, input, fake, pred_real, pred_fake)

        #################
        # Anomaly Score #
        #################
        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        batch_score, _ = torch.max(score, dim=0)

        # log
        self.log("val_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_score", batch_score.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)

    def generate_and_save_samples(self, epoch):
        # Generate and save example images
        self.generator.eval()  # Set the model to evaluation mode to ensure deterministic results
        with torch.no_grad():
            # Generate samples from your GAN
            input = pad_nextpow2(self.example_images)

            gen_image, _, _ = self.generator(input)

            # Convert generated samples to a grid for visualization (using torchvision)
            num_samples = self.example_images.size(0)
            grid = torchvision.utils.make_grid((gen_image + 1) / 2, nrow=int(num_samples**0.5))
            filename = f"GANomaly_fake_epoch={epoch}.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)
            
        self.generator.train()  # Set the model back to training mode 
    
    def on_validation_epoch_end(self):

        # Specify when to generate and save examples (e.g., every n epochs)
        if self.example_images is not None:

            if self.current_epoch % self.save_n_epochs == 0:

                # Call the method to generate and save examples
                self.generate_and_save_samples(self.current_epoch)
    
