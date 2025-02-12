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
            Shape '(n_samples, n_patches, embed_dim)'
        """
        x = self.proj(
            x
        ) # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)   # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)

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
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """ Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'

        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'
        """

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches +1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x

class MLP(nn.Module):
    """ Multilayer perceptron

    Parameters
    ----------
    in_features : int
        Number of input features

    hidden_features : int
        Number of nodes in the hidden layer

    out_features : int
        Number of output features

    p : float
        Dropout probability

    Attributes
    ----------
    fc : nn.Linear
        The First Linear layer.

    act : nn.GELU
        GELU activation function

    fc2 : nn.Linear
        The second linear layer

    drop : nn.Dropout
        Dropout layer
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """ Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape '(n_samples, n_patches + 1, in_features)'

        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_patches + 1, out_features)'
        """







