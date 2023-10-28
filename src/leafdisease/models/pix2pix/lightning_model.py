"""anomaLEAF: an unsupervised anomaly detection model based on colour reconstruction

"""

from __future__ import annotations

import os
import logging
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch import Tensor, optim
import torchvision

from leafdisease.utils.image import pad_nextpow2
from .loss import GeneratorLoss, DiscriminatorLoss
from leafdisease.components.pix2pix import Generator, Discriminator

logger = logging.getLogger(__name__)

class pix2pix(pl.LightningModule):
    """PL Lightning Module for the pix2pix .

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
        n_features: int,
        num_input_channels=3,
        num_output_channels=3,
        wadv: int = 1,
        wcon: int = 100,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        save_examples_every_n_epochs: int = 10,
        example_batch: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.generator: Generator = Generator(
            input_dim=num_input_channels,
            num_filter=n_features, 
            output_dim=num_output_channels
        )
        self.generator.normal_weight_init()

        self.discriminator: Discriminator = Discriminator(
            input_dim=num_output_channels+num_input_channels,
            num_filter=n_features,
            output_dim=1
        )
        self.discriminator.normal_weight_init()

        self.generator_loss = GeneratorLoss(wadv=wadv,wcon=wcon)
        self.discriminator_loss = DiscriminatorLoss()

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # for visualising GAN training
        self.save_n_epochs = save_examples_every_n_epochs
        self.example_batch = example_batch
        self.save_example_dir = save_example_dir

        # important for training with multiple optimizers
        self.automatic_optimization = False

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

        target = pad_nextpow2(batch["image"])
        
        input = torch.mean(target, dim=1, keepdim=True)

        ##########################
        # Optimize Discriminator #
        ##########################
        self.toggle_optimizer(disc_optimiser)
        with torch.no_grad():
            output = self.generator(input)

        pred_real = self.discriminator(input_tensor=input, target_tensor=target)
        pred_fake = self.discriminator(input_tensor=input, target_tensor=output)

        # loss
        disc_loss_fake, disc_loss_real, disc_loss = self.discriminator_loss(pred_real, pred_fake)

        # Discriminator grad calculations
        self.manual_backward(disc_loss)
        disc_optimiser.step()
        disc_optimiser.zero_grad()
        self.untoggle_optimizer(disc_optimiser)

        ######################
        # Optimize Generator #
        ######################
        self.toggle_optimizer(gen_optimiser)
        output = self.generator(input)

        with torch.no_grad():
            pred_fake = self.discriminator(input_tensor=input, target_tensor=output)

        # loss
        gen_adv_loss, gen_con_loss, gen_loss = self.generator_loss(patch_map=pred_fake, real=target, fake=output)

        # Generator grad calculations
        self.manual_backward(gen_loss)
        gen_optimiser.step()
        gen_optimiser.zero_grad()
        self.untoggle_optimizer(gen_optimiser)

        # Log
        self.log("train_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_disc_loss_fake", disc_loss_fake.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_disc_loss_real", disc_loss_real.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)

        self.log("train_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gen_adv_loss", gen_adv_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gen_con_loss", gen_con_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """

        target = pad_nextpow2(batch["image"])
        
        input = torch.mean(target, dim=1, keepdim=True)

        output = self.generator(input)

        ######################
        # Discriminator Loss #
        ######################
        pred_real = self.discriminator(input_tensor=input, target_tensor=target)
        pred_fake = self.discriminator(input_tensor=input, target_tensor=output)
        disc_loss_fake, disc_loss_real, disc_loss = self.discriminator_loss(pred_real, pred_fake)

        ##################
        # Generator Loss #
        ##################
        gen_adv_loss, gen_con_loss, gen_loss = self.generator_loss(patch_map=pred_fake, real=target, fake=output)

        # Log
        self.log("val_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_disc_loss_fake", disc_loss_fake.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_disc_loss_real", disc_loss_real.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)

        self.log("val_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_adv_loss", gen_adv_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_con_loss", gen_con_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)



    def generate_and_save_samples(self, epoch):
        # Generate and save example images
        self.generator.eval()  # Set the model to evaluation mode to ensure deterministic results
        with torch.no_grad():
            target = pad_nextpow2(self.example_batch["image"])
        
            input = torch.mean(target, dim=1, keepdim=True)

            output = self.generator(input)

            # Convert generated samples to a grid for visualization (using torchvision)
            num_samples = target.size(0)

            grid = torchvision.utils.make_grid(output, nrow=int(num_samples**0.5))
            filename = f"{epoch}-epoch.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)
            
        self.generator.train()  # Set the model back to training mode 
    
    def on_validation_epoch_end(self):

        # Specify when to generate and save examples (e.g., every n epochs)
        if self.example_batch is not None:

            if self.current_epoch % self.save_n_epochs == 0:

                # Call the method to generate and save examples
                self.generate_and_save_samples(self.current_epoch)