import torch
from torch import Tensor, nn

from leafdisease.models.components.autoencoder import Encoder, Decoder
from leafdisease.utils.image import generate_masks, pad_nextpow2


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
        latent_vec_features: int,
        latent_vec_dim: int,
        num_input_channels=3,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True, 
        k: int = 16
    )-> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_features=latent_vec_features,
            latent_vec_dim=latent_vec_dim,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
        )

        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

        self.mask = generate_masks(k_list=[k], n=batch_size, im_size=input_size[0], num_channels=num_input_channels)

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

    def mask_input(self, batch: Tensor) -> Tensor:
        # create masks
        list_mask = next(self.mask)
        assert(len(list_mask)==batch.size(0)) 

        grayscale = torch.mean(batch, dim=1, keepdim=True) # grayscale copy of batch
        grayscale = grayscale.repeat(1,3,1,1) # cast to three dimensions
        assert(grayscale.shape == batch.shape)
        mask_A_batch = batch.clone()
        mask_A_batch[list_mask==0] = grayscale[list_mask==0]
        assert(mask_A_batch.shape == batch.shape)
        mask_B_batch = batch.clone()
        mask_B_batch[list_mask==1] = grayscale[list_mask==1]
        assert(mask_B_batch.shape == batch.shape)

        return mask_A_batch, mask_B_batch, list_mask

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

        # reconstruct image
        fake = fake_A.clone()
        fake[mask==1] = fake_B[mask==1]
        assert(fake.shape == batch.shape)

        return padded, fake
      


class Discriminator(nn.Module):
    """Discriminator.

        PatchGAN discriminator which takes in image pairs, input and target, and calculates a
        patchwise anomaly score

    Args:
        input_size (tuple[int, int]): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
    """

    def __init__(
        self, input_size: tuple[int, int], num_input_channels: int, n_features: int
    ) -> None:
        super().__init__()

        # input layer
        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # encoder
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2     # use the smaller dimension
        while pyramid_dim > 32:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                 )
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Patch layer
        self.patch_layer = nn.Sequential()
        # self.patch_layer.add_module(f"patch-{out_features}-pad",
        #                                      nn.ZeroPad2d())
        self.patch_layer.add_module(f"patch-{n_features}-{512}-convt", 
                                             nn.Conv2d(n_features, 512, kernel_size=4, stride=1,
                                                padding=0,
                                                bias=False))
        self.patch_layer.add_module(f"patch-{out_features}-batchnorm",
                                             nn.BatchNorm2d(out_features)
                                             )
        self.patch_layer.add_module(f"patch-{out_features}-leakyrelu",
                                             nn.LeakyReLU(True)
                                             )
        # self.patch_layer.add_module(f"patch-{out_features}-pad",
        #                                      nn.ZeroPad2d())
        self.patch_layer.add_module(f"patch-{512}-{1}-convt", 
                                             nn.Conv2d(512, 1, kernel_size=4, stride=1,
                                                padding=0,
                                                bias=False))

    def forward(self, input_tensor: Tensor, target_tensor: Tensor) -> tuple[Tensor, Tensor]:
        """ Returns reconstruction patch map and classifier output """

        # generate a patchwise feature map
        input = torch.cat([input_tensor, target_tensor],dim=1)
        output = self.input_layers(input)
        output = self.pyramid_features(output)
        patch_map = self.patch_layer(output)

        return patch_map


class Generator(nn.Module):
    """Generator model.

    Made of an Unet architecture with skip links.

    Args:
        input_size (tuple[int, int]): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_features: int,
        latent_vec_dim: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_size, 
                               latent_vec_features=latent_vec_features, 
                               latent_vec_dim=latent_vec_dim, 
                               num_input_channels=num_input_channels, 
                               n_features=n_features, 
                               extra_layers=extra_layers, 
                               skips=True
                               )
        self.decoder = Decoder(input_size,  
                               latent_vec_dim=latent_vec_dim, 
                               latent_vec_features=latent_vec_features,
                               num_input_channels=num_input_channels, 
                               n_features=n_features, 
                               extra_layers=extra_layers, 
                               skips=True
                               )

    def forward(self, input_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return generated image"""
        latent_vector, skips = self.encoder(input_tensor)
        gen_image = self.decoder(latent_vector, skips)
        return gen_image
    
