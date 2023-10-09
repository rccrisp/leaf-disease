import math
from torch import Tensor, nn
import numpy as np

def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    padded_batch = nn.functional.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch

def generate_masks(k_list, n, im_size, num_channels=3):
    """Generates random tile masks for images

    Args:
        k_list (int): size of tiles
        n (int): number of permutations to generate at each tile size
        im_size (int): image dimensions (assumes square)
        num_channels (int): number of channels for mask

    Returns:
        A list of masks

    Adapated from code at https://github.com/plutoyuxie/Reconstruction-by-inpainting-for-visual-anomaly-detection/blob/main/utils/__init__.py
    
    """

    while True:
        Ms = []
        for k in k_list:
            N = im_size // k
            rdn = np.random.permutation(N**2)
            additive = N**2 % n
            if additive > 0:
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive))))
            n_index = rdn.reshape(n, -1)
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N**2)]
                tmp = np.asarray(tmp).reshape(N, N)
                tmp = tmp.repeat(k, 0).repeat(k, 1)
                # Create a 3D mask with 'num_channels' channels
                mask_3d = np.stack([tmp] * num_channels, axis=-1)
                Ms.append(mask_3d)
        yield Ms