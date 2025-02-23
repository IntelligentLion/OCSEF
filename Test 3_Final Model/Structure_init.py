import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sympy.abc import alpha
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
from ViT import VisionTransformer

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
NET_INPUT_SIZE = (572, 572)
def Shape_Process(img_tensor):
    """
    Processes the input image tensor to match the network input size using PyTorch operations.

    Args:
        img_tensor (torch.Tensor): Input image tensor in CxHxW format.
        NET_INPUT_SIZE (tuple): Target spatial size (height, width).

    Returns:
        torch.Tensor: Processed image tensor.
    """
    _, C, H, W = img_tensor.shape
    TH, TW = NET_INPUT_SIZE

    # Check if the image is larger than the target size in both dimensions
    if H > TH and W > TW:
        # Resize to target size using bicubic interpolation
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=NET_INPUT_SIZE,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
    # Check if the image is smaller than the target size in both dimensions
    elif H < TH and W < TW:
        # Resize if height is odd
        if H % 2 != 0:
            new_H = H // 2
            new_W = int(W * (new_H / H))  # Maintain aspect ratio
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(new_H, new_W),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            H, W = new_H, new_W

        # Calculate padding (handling possible odd dimensions)
        pad_h = (TH - H + 1) // 2  # Ensures total padding adds up correctly
        pad_w = (TW - W + 1) // 2
        # Apply reflection padding
        img_tensor = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    return img_tensor

def feature_addition(x, attention_map, stage):
    B, C, H, W = x.size()
    x = x.view(B, C, H * W).permute(2, 0, 1)  # Reshape to (H*W, B, C)

        # Extract attention features for the current stage
    attn_features = attention_map[stage - 1, :, :]  # Shape: (256, 784)

        # Perform matrix multiplication with attention features
    weighted_x = torch.matmul(attn_features.permute(1, 0), x.permute(1, 0, 2))

    weighted_x = weighted_x.permute(1, 2, 0).view(B, C, H, W)

    return weighted_x

# Model: UNet architecture (simplified)
class UNetViTCross(nn.Module):
    def __init__(self):
        super(UNetViTCross, self).__init__()

        self.cross_model = VisionTransformer(in_chans = 1)
        self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleConv_1 = doubleConv2d(1, 64)
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

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def feature_addition(x, attention_map, stage):
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(2, 0, 1)  # Reshape to (H*W, B, C)

        # Extract attention features for the current stage
        attn_features = attention_map[stage - 1, :, :]  # Shape: (256, 784)

        # Perform matrix multiplication with attention features
        weighted_x = torch.matmul(attn_features.permute(1, 0), x.permute(1, 0, 2))

        weighted_x = weighted_x.permute(1, 2, 0).view(B, C, H, W)

        return weighted_x


    def forward(self, image, feature_tensor):
        #t_embed = torch.rand_like(image) * t.view(-1, 1, 1, 1)
        #image = torch.cat([image, t_embed], dim=1)
        #print(image.size())

        # encoder
        x1 = self.doubleConv_1(image) ##
        x2 = self.maxPool_2x2(x1)
        x3 = self.doubleConv_2(x2) ##
        x4 = self.maxPool_2x2(x3)
        x5 = self.doubleConv_3(x4) ##
        x6 = self.maxPool_2x2(x5)
        x7 = self.doubleConv_4(x6) ##
        x8 = self.maxPool_2x2(x7)

        x9 = self.doubleConv_5(x8) # bottleneck
        x_i = self.cross_model(x9, feature_tensor)

        # apply the cross
        #x = self.cross_model(x9)

        # decoder and feature addition
        x9 = feature_addition(x9, x_i, 1)
        x_up_1 = self.up_trans_1(x9)
        y_1 = crop_img(x7, x_up_1)
        #y_1 = self.cross_model(y_1) # Apply cross attention
        z1 = torch.cat([x_up_1, y_1], 1)
        x_cat_1 = self.up_conv_1(z1)

        x_cat_1 = feature_addition(x_cat_1, x_i, 2)
        x_up_2 = self.up_trans_2(x_cat_1)
        y_2 = crop_img(x5, x_up_2)
        z2 = torch.cat([x_up_2, y_2], 1)
        x_cat_2 = self.up_conv_2(z2)  ######################

        x_cat_2 = feature_addition(x_cat_2, x_i, 3)
        x_up_3 = self.up_trans_3(x_cat_2)
        y_3 = crop_img(x3, x_up_3)
        z3 = torch.cat([x_up_3, y_3], 1)
        x_cat_3 = self.up_conv_3(z3)

        x_cat_3 = feature_addition(x_cat_3, x_i, 4)
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
    def __init__(self,
                 in_chans = 1024,
                 img_size = 512,
                 patch_size = 16,
                 embed_dim = 128):
        #declaring basic structure for cross
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans = in_chans,
                                      embed_dim = 128)
        self.attn_layer = nn.MultiheadAttention(embed_dim=768, num_heads=4)
        self.projection = nn.Linear(768, 1024)

    def forward(self, x):
        b, c, h, w, = x.shape
        x_patches = self.patch_embed(x) # converting the input to patches
        attn_output, n = self.atten_layer(x_patches, x_patches, x_patches)
        attn_output = self.projection(attn_output).permute(0, 2, 1).view(b,c,h,w) #remake shape
        return (x + attn_output) / (1**0.5) # adding the features to UNET and scaling feature intensity using temperature calculation

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
    def __init__(self, input, dim, num_heads=16, ):
        super().__init__()
        self.input = input
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Query projection for input x
        self.query = nn.Linear(input.size(), input.size())
        # Key/Value projections for context
        self.key = nn.Linear(input.size(), input.size())
        self.value = nn.Linear(input.size(), input.size())

        # Output projection
        self.out_proj = nn.Linear(input.size(), input.size())

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


x = torch.randn([1,1,572,572])
model = UNet().to('cuda' if torch.cuda.is_available() else 'cpu')
model(x)