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

from leafdisease.utils.image import pad_nextpow2
from .model import ganomalyModel
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
        save_examples_every_n_epochs: int = 10,
        example_images: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.model: ganomalyModel = ganomalyModel(
            input_size=input_size,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer
        )

        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
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

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.model.generator.parameters(),
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

        ##########################
        # Optimize Discriminator #
        ##########################
        padded, fake, _, _ = self.model(batch["image"])

        pred_real, _ = self.model.discriminator(padded)
        pred_fake, _ = self.model.discriminator(fake)

        # loss
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        # Discriminator grad calculations
        disc_optimiser.zero_grad()
        self.manual_backward(disc_loss)
        disc_optimiser.step()

        ######################
        # Optimize Generator #
        ######################
        padded, fake, latent_i, latent_o = self.model(batch["image"])

        _, feature_real = self.model.discriminator(padded)
        _, feature_fake = self.model.discriminator(fake)
        
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

        padded, fake, latent_i, latent_o = self.model(batch["image"])

        # calculate the anomaly score
        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)
        batch_score = torch.mean(score)

        pred_real, _ = self.model.discriminator(padded)

        pred_fake, _ = self.model.discriminator(fake.detach())
        disc_loss = self.discriminator_loss(pred_real, pred_fake)
        
        pred_fake, _ = self.model.discriminator(fake)
        gen_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)

        # log
        self.log("val_score", batch_score.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
    
    def predict_step(self, batch: dict[str, str | Tensor], batch_idx):

        del batch_idx
        
        # validation step is used for inference
        padded, fake, latent_i, latent_o = self.model(batch["image"])

        score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)

        return {"real": padded, "generated": fake, "anomaly_score": score, "filename": batch["filename"]}

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
    
