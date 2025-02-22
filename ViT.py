import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fontTools.varLib.models import VariationModel
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

    def __init__(self):
        super(PatchEmbed, self).__init__()
        self.img_size = 512
        self.patch_size = 32
        self.in_chans = 1
        self.embed_dim = 768
        self.n_patches = (self.img_size // self.patch_size) ** 2   # n_patches = (img_size//patch_size)^2

        self.proj = nn.Conv2d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
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
        '''Put whatever goes here'''
        #x = F.interpolate(x, size = (28,28), mode = 'bilinear', align_corners = False)

        return x



# Access image in training loop
# Split into patches
# Iterate through each patch
# Resize each patch to (28, 28)
# Input patch and context (x9) from bottleneck into DualInputCrossAttention module

class DualInputCrossAttention(nn.Module):
    def __init__(self, input1, context, num_heads, drop_p=0.2):
        super().__init__()

        self.num_heads = num_heads

        # First set of patches
        self.input1 = input1
        self.dim1 = input1.size(3) # Patch size: (28, 28)

        # Second set of patches
        self.context = context  # bottleneck
        self.dim2 = context.size(3) # Patch size: (28, 28) from bottleneck

        if self.dim1 != self.dim2:
            print("Input and Context are of different dimensions")
            raise ValueError
        if not (self.dim1 % self.num_heads == 0 and self.dim2 % self.num_heads == 0):    # dim1 and dim2 must be divisible by num_head
            print("Input and Context are not divisible by num_heads")
            raise ValueError

        # Query projection for input x
        self.query = nn.Linear(self.dim1, self.dim1)
        # Key/Value projections for context
        self.key = nn.Linear(self.dim2, self.dim2)
        self.value = nn.Linear(self.dim2, self.dim2)

        # Output projection
        self.out_proj = nn.Linear(self.dim1, self.dim1)

        self.proj_drop = nn.Dropout(drop_p)

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
        attended = self.out_proj(attended)
        return self.proj_drop(attended)

class Block(nn.Module):
    """ Transformer block

    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads
    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'
    qkv_bias : bool
        If True then we include bias to query, key, and value projections
    p, attn_p : float
        Dropout probability

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attenion
        Attention module

    mlp : MLP
        MLP module.
    """

    def __init__(self, embed_dim, n_heads=16, qkv_bias = True, p=0., attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=p, bias = qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps = 1e-6)


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

        x = x + self.attn(self.norm1(x))
        x = x + self.norm2(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 input_1,
                 context,
                 in_chans = 1,
                 embed_dim = 768,
                 depth = 2,
                 n_heads = 16,
                 qkv_bias = True,
                 drop_p=0.2,
                 attn_p = 0.):

        super().__init__()

        self.input_1 = input_1
        self.context = context
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_p)

        self.block =  Block( embed_dim = embed_dim,
                              n_heads = n_heads,
                              qkv_bias = qkv_bias,
                              p = drop_p,
                              attn_p = attn_p
            )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, 1)

        self.linear_proj = self.patch_embed(x)

        self.dual_cross_attention = DualInputCrossAttention(input1=self.input_1, context=context, num_heads=n_heads)

    def forward(self, x, context):
        #n_samples = x.shape[0] # (n_samples, in_chans, img_size, img_size)
        x_embed = self.patch_embed(x)
        final = []
        for i in range(len(x_embed)):
            input1 = x_embed[i]
            input1 = Block(input1)
            input = DualInputCrossAttention(input1, context, num_heads=n_heads)
            x = input + self.pos_embed #(n_samples, 1 + n_patches, embed_dim)
            x = self.pos_drop(x)
            x = self.norm(x)
            final.append(x)
        return final

x = torch.randn([1, 1, 512, 512])
context = torch.randn([1, 1, 512, 512])
model = VisionTransformer(input_1=x, context=context)
result = model()
print(result)

""" 
class CrossAttention(nn.Module):
    def __init__(self, img_size=512, patch_size=32, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.scale = head_dim ** -0.5
        self.heads = heads
        self.head_dim = head_dim
        self.dropout = nn.Dropout(p=dropout)

        self.qW = nn.Linear(img_size, img_size, bias = False)
        self.kWvW = nn.Linear(img_size, img_size * 2, bias = False)

        self.out = nn.Sequential(
            nn.Linear(img_size, img_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        b, n, _, h = *x.shape, self.heads
        h = self.heads
        x = self.norm(x) 
"""

data_dir = '/Users/humza/PycharmProjects/OCSEF/dataset/ICH_train'
dicom_files = sorted(glob(os.path.join(data_dir, "*.dcm")))

def load_dicom_to_tensor(file_path):
    # Load DICOM file
    dicom = pydicom.dcmread(file_path)

    # Print some basic metadata to verify correct reading
    print(f"Processing file: {file_path}")
    print(f"Patient ID: {dicom.PatientID if 'PatientID' in dicom else 'N/A'}")
    print(f"Modality: {dicom.Modality if 'Modality' in dicom else 'N/A'}")

    # Get pixel array and convert to float32
    image = dicom.pixel_array.astype(np.float32)

    # Ensure 2D image is in (H,W) format
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]  # Add channel dimension: (1,H,W)
    elif len(image.shape) == 3:
        image = image[0:1, ...]  # Take first channel if multi-channel

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(np.array(image).astype('float32'))
    return tensor



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 3e-4
num_epochs = 1
img_size=28
patch_size=16
in_chans=1
embed_dim=128
depth=12
n_heads=12
mlp_ratio=4
qkv_bias=True
p=0.
attn_p=0.

input_1 = torch.randn([1, 1, 16, 16])
context = torch.randn([1,1,16,16])
# Initialize model
model = VisionTransformer(input_1 = input_1,
                          context = context,
                          img_size = img_size,
                          patch_size = patch_size,
                          in_chans = in_chans,
                          embed_dim = embed_dim,
                          depth = depth,
                          n_heads = n_heads,
                          qkv_bias = qkv_bias,
                          drop_p=p,
                          attn_p = attn_p).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
loss = 0
for i in tqdm(range(len(dicom_files))):
    tensor = load_dicom_to_tensor(dicom_files[i])
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    x = model(tensor)
    print(x)