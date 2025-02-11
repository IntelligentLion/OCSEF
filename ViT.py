import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.utils import save_image
from torch import optim
from tqdm import tqdm
import pandas as pd
import os
import argparse
import csv

class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.

    Parameters
    ---------
    img_size : int
        Size of the image (it is a square).

    patch_size: int
        Size of the patch (it is a square).

    in_chans: int
        Number of input channels.

    embed_dim : int
        The embedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) * 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        ----------
        x = torch.Tensor
            Shape '(n_samples, in_chans, img_size, img_size)'

        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_patches, embed_dim)
        """
        x = self.proj(
            x
        )
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

class Attention(nn.Module):
    """"
    Attention mechanism.

    Parameters
    ----------

    dim: int
        The input and output dimension of per token features
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key, and value projections
    attn_p: float
        Dropout probability applied to the query, key, and value tensors.
    proj_p : float
        Dropout probability applied to the output tensors

    Atributes
    ---------

    scale : float
        Normalizing constant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key, and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attenetion heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

