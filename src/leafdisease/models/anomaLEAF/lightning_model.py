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
        batch_size: int,
        input_size: tuple[int, int],
        n_features: int,
        mask_size: int = 16,
        num_input_channels=3,
        wadv: int = 1,
        wcon: int = 100,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        save_examples_every_n_epochs: int = 10,
        example_images: Tensor = None,
        save_example_dir: str = "examples"
    ) -> None:
        super().__init__()

        self.model: anomaleafModel = anomaleafModel(
            batch_size=batch_size,
            input_size=input_size,
            n_features=n_features,
            num_input_channels=num_input_channels,
            k=mask_size
        )

        self.generator_loss = GeneratorLoss(wadv=wadv,wcon=wcon)
        self.discriminator_loss = DiscriminatorLoss()

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

        # for visualising GAN training
        self.save_n_epochs = save_examples_every_n_epochs
        self.example_images = example_images
        self.save_example_dir = save_example_dir

        self.training_reconstruction_loss = float('-inf')

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
        output = self.model(batch["image"])
        # mask
        pred_real_A = self.model.discriminator(input_tensor=output["input_A"], target_tensor=output["real"])
        pred_fake_A = self.model.discriminator(input_tensor=output["input_A"], target_tensor=output["fake_A"])
        # loss
        disc_loss_A = self.discriminator_loss(pred_real_A, pred_fake_A)

        # inv mask
        pred_real_B = self.model.discriminator(input_tensor=output["input_B"], target_tensor=output["real"])
        pred_fake_B = self.model.discriminator(input_tensor=output["input_B"], target_tensor=output["fake_B"])
        # loss
        disc_loss_B = self.discriminator_loss(pred_real_B, pred_fake_B)
        
        # total disc loss
        disc_loss = disc_loss_A + disc_loss_B

        # Discriminator grad calculations
        disc_optimiser.zero_grad()
        self.manual_backward(disc_loss)
        disc_optimiser.step()

        ######################
        # Optimize Generator #
        ######################
        output = self.model(batch["image"])
        pred_fake_A = self.model.discriminator(input_tensor=output["input_A"], target_tensor=output["fake_A"])
        pred_fake_B = self.model.discriminator(input_tensor=output["input_B"], target_tensor=output["fake_B"])
        # loss
        gen_loss_A = self.generator_loss(patch_map=pred_fake_A, real=output["real"], fake=output["fake_A"])
        gen_loss_B = self.generator_loss(patch_map=pred_fake_B, real=output["real"], fake=output["fake_B"])

        gen_loss = gen_loss_A + gen_loss_B

        # Generator grad calculations
        gen_optimiser.zero_grad()
        self.manual_backward(gen_loss)
        gen_optimiser.step()

        #################################
        # Adaptive Threshold Classifier #
        #################################
        # reconstruction loss
        heatmap = torch.abs(output["real"] - output["fake_B"])
        heatmap = torch.sum(heatmap, dim=1, keepdim=True)
        score, _ = self.model.classifier(heatmap, self.model.threshold)
        max_score, _ = torch.max(score, dim=0)
        if self.training_reconstruction_loss < max_score:
            self.training_reconstruction_loss = max_score

        # Log
        self.log("train_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("train_score", max_score, on_step=False, on_epoch=True, prog_bar=True, logger=self.logger )

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}
    
    def on_train_epoch_end(self) -> None:
        """During training, update the classifier threshold with the expected loss for normal samples
        """
        output = super().on_train_epoch_end()

        if self.training_reconstruction_loss < self.model.threshold:
            self.model.threshold = self.training_reconstruction_loss
        
        self.training_reconstruction_loss = float("-inf")

        return output

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) :
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """

        # model output
        padded = pad_nextpow2(batch["image"])

        # create masks
        masked, _, _ = self.model.mask_input(padded)
        fake = self.model.generator(masked)
        
        ##########################
        # Evaluate Discriminator #
        ##########################
        # mask
        pred_real = self.model.discriminator(input_tensor=masked, target_tensor=padded)
        pred_fake = self.model.discriminator(input_tensor=masked, target_tensor=fake)

        # loss
        disc_loss = self.discriminator_loss(pred_real, pred_fake)

        ######################
        # Evaluate Generator #
        ######################
        
        # loss
        gen_loss = self.generator_loss(patch_map=pred_fake, real=padded, fake=fake)

        #################################
        # Adaptive Threshold Classifier #
        #################################
        # reconstruction loss
        heatmap = torch.abs(padded - fake)
        heatmap = torch.sum(heatmap, dim=1, keepdim=True)
        score, _ = self.model.classifier(heatmap, self.model.threshold)
        max_score, _ = torch.max(score, dim=0)

        # log
        self.log("val_disc_loss", disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_gen_loss", gen_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=self.logger)
        self.log("val_score", max_score, on_step=False, on_epoch=True, prog_bar=True, logger=self.logger )
    
    def predict_step(self, batch: dict[str, str | Tensor], batch_idx):

        del batch_idx
        
        # validation step is used for inference
        output = self(batch["image"])
        output["filename"] = batch["filename"]

        return output

    def generate_and_save_samples(self, epoch):
        # Generate and save example images
        self.eval()  # Set the model to evaluation mode to ensure deterministic results
        with torch.no_grad():
            # Generate samples from your GAN
            generated_samples = self.model(self.example_images)

            # Convert generated samples to a grid for visualization (using torchvision)
            num_samples = self.example_images.size(0)
            real = generated_samples["real"]
            grid = torchvision.utils.make_grid(real, nrow=int(num_samples**0.5))
            filename = f"anomaLEAF_real_epoch={epoch}.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)

            fake = generated_samples["fake"]
            grid = torchvision.utils.make_grid(fake, nrow=int(num_samples**0.5))
            filename = f"anomaLEAF_fake_epoch={epoch}.png"
            save_path = os.path.join(self.save_example_dir, filename)
            torchvision.utils.save_image(grid, save_path)
            print(save_path)
            
        self.train()  # Set the model back to training mode 
    
    def on_validation_epoch_end(self):

        # Specify when to generate and save examples (e.g., every n epochs)
        if self.example_images is not None:

            if self.current_epoch % self.save_n_epochs == 0:

                # Call the method to generate and save examples
                self.generate_and_save_samples(self.current_epoch)