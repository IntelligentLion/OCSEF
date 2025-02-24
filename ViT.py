import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fontTools.varLib.models import VariationModel
from sympy import python
from torchvision.transforms import v2
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.utils import save_image
from torch import optim
from tqdm import tqdm
import os
import argparse
import medmnist
import pandas as pd
import pydicom
import numpy as np
from glob import glob
import cv2 as cv


NET_INPUT_SIZE = (572, 572)

class PatchEmbed(nn.Module):
    """
    Split image into patches and then embed them.
    """

    def __init__(self, img_size=572, patch_size=22, in_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Run forward pass.
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)   # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)

        return x


# Access image in training loop
# Split into patches
# Iterate through each patch
# Resize each patch to (28, 28)
# Input patch and context (x9) from bottleneck into DualInputCrossAttention module

class DualInputCrossAttention(nn.Module):
    def __init__(self, num_heads=16, embed_dim = 1024):
        super().__init__()


        # Query projection for input x
        self.query = nn.Linear(embed_dim, embed_dim)
        # Key/Value projections for context
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )



    def forward(self, x, context):

        # Project inputs
        B, C, _, _ = context.size()
        context = context.view(B, C, -1)
        context = context.permute(0, 2, 1)  # (B, Lx, D)
        print(context.shape)
        print(x.shape)

        q = self.query(x)  # (B, Lx, D)
        k = self.key(context)  # (B, Lctx, D)
        v = self.value(context)  # (B, Lctx, D)

        # Initialize Multihead Attention
        mha = nn.MultiheadAttention(embed_dim=1024, num_heads=16, dropout=self.drop_p, bias=False)

        # Compute attention scores
        attn_map, _ = mha(q, k, v)
        print(attn_map.shape)

        return attn_map

class Block(nn.Module):

    def __init__(self, embed_dim, n_heads=16, qkv_bias=True, p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm([1, 676, 1024], eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=n_heads, dropout=p, bias=qkv_bias, batch_first=True)
        self.attn_weights = None


    def forward(self, x):
        """Runs forward pass.

        Paarameters
        -----------
        x : Torch.tensor()
            Shape '(n_samples, n_patches + 1, dim)'

        Returns
        -------
        torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'

        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        self.attn_weights = self.attn.in_proj_weight

        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 in_chans = 1,
                 embed_dim = 768,
                 depth = 2,
                 n_patches = 256,
                 n_heads = 16,
                 qkv_bias = True,
                 drop_p=0.2,
                 attn_p = 0.):

        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros((1, 676, 1024))
        )
        self.pos_drop = nn.Dropout(p=drop_p)

        self.block = Block(embed_dim = 1024,
                           n_heads = n_heads,
                           qkv_bias = qkv_bias,
                           p = drop_p,
       )


        self.norm = nn.LayerNorm([1, 676, 1024], eps=1e-6)
        self.head = nn.Linear(1024, 1)

        self.linear_proj = PatchEmbed()
        self.n_heads = n_heads

    def forward(self, x, context):
        #n_samples = x.shape[0] # (n_samples, in_chans, img_size, img_size)
        x_embed = self.linear_proj(x)
        print("x_embed: ", x_embed)
        final = []
        for i in range(26):
            input = x_embed[:, i, :]
            print(input)
            input = Block(input)
            print("input after block: ", input)
            input = DualInputCrossAttention(input, context)

            #x = torch.cat([input, self.pos_embed], dim=1)
            x = input + self.pos_embed #(n_samples, 1 + n_patches, embed_dim)
            x = self.pos_drop(x)
            x = self.norm(x)
            final.append(x)
        return final