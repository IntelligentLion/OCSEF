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

        # Compute attention scores
        attn_map, _ = self.mha(q, k, v)
        print(attn_map.shape)

        return attn_map

class Block(nn.Module):

    def __init__(self, embed_dim=1024, n_heads=16, qkv_bias=True, drop_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=drop_p, bias=qkv_bias, batch_first=True)

    def forward(self, x):
        """Runs forward pass.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        self.attn_weights = self.attn.in_proj_weight

        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 in_chans = 1,
                 embed_dim = 1024,
                 depth = 2,
                 n_patches = 676,
                 n_heads = 16,
                 qkv_bias = True,
                 drop_p=0.1,
                 attn_p = 0.1):

        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 676, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_p)

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, n_heads=n_heads, qkv_bias=qkv_bias, drop_p=drop_p)
            for _ in range(depth)
        ])


        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, context):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        #x = self.head(x.mean(dim=1))

        return x

if __name__ == "__main__":
    model = VisionTransformer()
    x = torch.randn(1, 1, 572, 572)  # Example input
    context = torch.randn(1, 1, 572, 572)  # Example context
    output = model(x, context)
    print(output.shape)  # Should print: torch.Size([1, 1])