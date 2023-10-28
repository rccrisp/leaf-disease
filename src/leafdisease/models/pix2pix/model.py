import torch
from torch import nn, Tensor

from leafdisease.utils.image import pad_nextpow2, generate_masks



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
        anomaly_size: int = 4,
        k: int = 16,
        threshold: float = float('-inf')
    )-> None:
        super().__init__()
       

        self.mask_gen = generate_masks(k_list=[k], n=batch_size, im_size=input_size[0], num_channels=num_input_channels)

        self.threshold = threshold

    def mask_input(self, batch: Tensor) -> Tensor:
        # create masks
        mask = next(self.mask_gen)
        assert mask[0].max() == 1, f"No input was masked ({mask[0].min()},{mask[0].max()})"
        mask = mask.to(batch.device)
        mask = mask[:batch.shape[0]]
        assert mask.device == batch.device, f"Different devices: masks ({mask.device} batch ({batch.device}))"

        # grayscale
        grayscale = torch.mean(batch, dim=1, keepdim=True) # grayscale copy of batch
        grayscale = grayscale.repeat(1,3,1,1) # cast to three dimensions

        # created reduce image
        replace_mask = mask.eq(0)
        copy = batch.clone()
        assert grayscale.device == copy.device, f"Different devices: grayscale ({grayscale.device}) copy ({copy.device})"
        assert grayscale.size() == copy.size(), f"grayscale shape ({grayscale.size()}) does not match original shape ({copy.size()})"
        mask_A_batch = torch.where(replace_mask, grayscale, copy)
        assert mask_A_batch.size() == copy.size(), f"mask shape ({mask_A_batch.size()}) does not match original shape ({copy.size()})"

        replace_mask_inv = mask.eq(1)
        copy = batch.clone()
        assert grayscale.size() == copy.size(), f"grayscale shape ({grayscale.size()}) does not match original shape ({copy.size()})"
        mask_B_batch = torch.where(replace_mask_inv, grayscale, copy)
        assert mask_B_batch.size() == copy.size(), f"mask shape ({mask_B_batch.size()}) does not match original shape ({copy.size()})"

        return mask_A_batch, mask_B_batch, mask, grayscale

    def forward(self, batch: Tensor):
        """Get scores for batch
        
        Args:
            batch (Tensor): Images
        
        Returns: 
            Tensor: regeneration score
        """
        padded = pad_nextpow2(batch)

        leaf_segment = (padded != 0).float()

         # create masks
        mask_A, mask_B, mask, grayscale = self.mask_input(padded)

        # regenerate from masks
        fake_A = self.generator(mask_A)
        fake_B = self.generator(mask_B)

        # reconstruct image
        fake = fake_A.clone()
        replace_mask = mask.eq(0)
        fake = torch.where(replace_mask, fake_A, fake_B)

        # when training we will evaluate only on one mask
        if self.training:
            fake = fake * leaf_segment
            return {"real": padded, "input": grayscale, "fake": fake}
        # when predicting, we will regenerate the whole image
        else:
            assert fake.size() == batch.size(), f"generated image ({fake.size()}) does not match original image ({batch.size()})"
            
            # score
            heatmap = torch.abs(fake - padded)
            heatmap = torch.sum(heatmap, dim=1, keepdim=True)
            score = self.classifier(heatmap)
            assert score.size()[0] == heatmap.size()[0], f"Score ({score.size()[0]}) does not match expected batch size ({heatmap.size()[0]})"
            label = self.threshold < score
            
            fake = fake * leaf_segment

            return {"real": padded, "fake": fake, "pred_score": score, "pred_label": label}
      
