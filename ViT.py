import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fontTools.varLib.models import VariationModel
from torchvision.transforms import v2
from torchvision.datasets.folder import default_loader
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
from medmnist import OrganAMNIST
from medmnist.info import INFO
import pandas as pd
import pydicom
import numpy as np

from test1.test_VAE1 import VariationalAutoEncoder


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
        Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.
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

        x = self.fc1 (
            x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)

        return x

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

    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p=0., attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads = n_heads,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = p
        )
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        hidden_features = int(mlp_ratio * dim)
        self.MLP = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

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
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size = 384, patch_size = 16, in_chans = 3, n_classes = 1000, embed_dim = 128, depth = 12, n_heads = 12, mlp_ratio = 4, qkv_bias = True, p = 0., attn_p = 0.):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(

        [
            Block(
                dim = embed_dim,
                n_heads = n_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                p = p,
                attn_p = attn_p
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) #(n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) #(n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed #(n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:0] # Just the CLS token
        x = self.head(cls_token_final)

        return x

class CustomCrossAttention(nn.Module):
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

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = torch.tensor(sample.drop('label').values, dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        if self.transform:
            features = self.transform(features)
        return features, label

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
patch_size=7
in_chans=1
embed_dim=128
depth=12
n_heads=12
mlp_ratio=4
qkv_bias=True
p=0.
attn_p=0.

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])



# Initialize model
model = VisionTransformer(img_size = img_size, patch_size = patch_size, in_chans = in_chans, n_classes = n_classes, embed_dim = embed_dim, depth = depth, n_heads = n_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, p = p, attn_p = attn_p).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

# Test the model
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)

        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

test_acc = 100. * test_correct / test_total
print(f"\nTest Results | Loss: {test_loss / len(test_loader):.4f} | Acc: {test_acc:.2f}%")
        



"""














