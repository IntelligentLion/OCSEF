import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.abc import alpha
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import random_split
import math
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob

def crop_img(tensor, target_tensor):
    tensor_size = tensor.size()[2] # (bs, c, h, w)
    target_size = target_tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

def doubleConv2d(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU()
    )
    return conv

# Model: UNet architecture (simplified)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleConv_1 = doubleConv2d(2, 64)
        self.doubleConv_2 = doubleConv2d(64, 128)
        self.doubleConv_3 = doubleConv2d(128, 256)
        self.doubleConv_4 = doubleConv2d(256, 512)
        self.doubleConv_5 = doubleConv2d(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = doubleConv2d(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = doubleConv2d(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = doubleConv2d(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = doubleConv2d(128, 64)

        self.out = nn.Conv2d(64, 2, kernel_size=1)
        self.cross_model = UNetViTCross() #connect cross to model


    def forward(self, image, t):
        t_embed = torch.rand_like(image) * t.view(-1, 1, 1, 1)
        image = torch.cat([image, t_embed], dim=1)
        print(image.size())
        # encoder
        x1 = self.doubleConv_1(image) ##
        x2 = self.maxPool_2x2(x1)
        x3 = self.doubleConv_2(x2) ##
        x4 = self.maxPool_2x2(x3)
        x5 = self.doubleConv_3(x4) ##
        x6 = self.maxPool_2x2(x5)
        x7 = self.doubleConv_4(x6) ##
        x8 = self.maxPool_2x2(x7)
        x9 = self.doubleConv_5(x8)

        # apply the cross
        x9 = self.cross_attention(x9)

        # decoder
        x_up_1 = self.up_trans_1(x9)
        y_1 = crop_img(x7, x_up_1)
        z1 = torch.cat([x_up_1, y_1], 1)
        x_cat_1 = self.up_conv_1(z1)

        x_up_2 = self.up_trans_2(x_cat_1)
        y_2 = crop_img(x5, x_up_2)
        z2 = torch.cat([x_up_2, y_2], 1)
        x_cat_2 = self.up_conv_2(z2)  ######################

        x_up_3 = self.up_trans_3(x_cat_2)
        y_3 = crop_img(x3, x_up_3)
        z3 = torch.cat([x_up_3, y_3], 1)
        x_cat_3 = self.up_conv_3(z3)

        x_up_4 = self.up_trans_4(x_cat_3)
        y_4 = crop_img(x1, x_up_4)
        z4 = torch.cat([x_up_4, y_4], 1)
        x_cat_4 = self.up_conv_4(z4)

        output = self.out(x_cat_4)
        return output

#ViT architecutre

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
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
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

#single cross
class UNetViTCross(nn.Module):
    def __init__(self):
        #declaring basic structure for cross
        super(UNetViTCross, self).__init__()
        self.vit = PatchEmbed(img_size=512, patch_size=4)
        self.attn_layer = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.projection = nn.Linear(768, 1024)

    def forward(self, x):
        b, c, h, w, = x.shape
        x_patches = self.patch_embed(x) # converting the input to patches
        attn_output, n = self.atten_layer(x_patches, x_patches, x_patches)
        attn_output = self.projection(attn_output).permute(0, 2, 1).view(b,c,h,w) #remake shape
        return x + attn_output # adding the featurees to UNET

'''class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads = 4, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Single input projection (since Q, K, V all come from the same source)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Project input to query, key, value
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)  # (B, N, D)
        v = self.v_proj(x)  # (B, N, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = attn_weights @ v  # (B, H, N, d)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (B, N, D)

        # Final projection
        output = self.out_proj(output)
        return output
'''

## use this one along with q for duel
class DualInputCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Query projection for input x
        self.query = nn.Linear(dim, dim)
        # Key/Value projections for context
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        """
        Args:
            x: (batch_size, seq_len_x, dim)
            context: (batch_size, seq_len_ctx, dim)

        Returns:
            attended: (batch_size, seq_len_x, dim)
        """
        batch_size = x.size(0)

        # Project inputs
        q = self.query(x)  # (B, Lx, D)
        k = self.key(context)  # (B, Lctx, D)
        v = self.value(context)  # (B, Lctx, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)

        # Concatenate heads and project back
        attended = attended.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.dim)
        return self.out_proj(attended)