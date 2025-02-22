import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
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

betas_total = torch.linspace(0.0001, 0.02, 1000) # all the betas within the specified range
alphas_total = 1. - betas_total # all the alphas within the range

# Diffusion Model Helper Functions
def noise_schedule(t):
    """Returns the beta_t and alpha_t_bar for a given time step."""
    beta_t = betas_total[t-1]
    alpha_t = alphas_total[t-1]

    alpha_t_bar = torch.cumprod(alpha_t, dim=0)  # Cumulative product for alpha_t_bar
    return beta_t, alpha_t_bar


def forward_diffusion(x_0, t):
    """Applies the forward diffusion process by adding noise to the image."""
    noise = torch.randn_like(x_0)
    alpha_t_bar = alphas_total[t-1]

    # Ensure alpha_t_bar is selected per batch element
    alpha_t_bar = alpha_t_bar.view(-1, 1, 1, 1)  # Proper broadcasting for each batch item
    x_t = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1 - alpha_t_bar) * noise
    return x_t, noise


def reverse_diffusion_step(x_t, t, model, betas):
    """Single reverse step during diffusion"""
    beta_t, alpha_t_bar = noise_schedule(t)
    noise_pred = model(x_t, t)
    return (x_t - beta_t * noise_pred) / torch.sqrt(alpha_t_bar)

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



#batch = torch.stack(tensors)  # Shape: [batch_size, 1, H, W]
#print(f"Batch shape: {batch.shape}")
#batch = (batch - batch.min()) / (batch.max() - batch.min())
#print(f"Normalized range: [{batch.min():.3f}, {batch.max():.3f}]")

# Hyperparameters
epochs = 1
batch_size = 2
lr = 2e-4
T = 5 # Number of timesteps in the forward diffusion process
best_loss = float('inf')  # Initialize best loss with infinity

# Model Checkpoint Path
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


model = UNet().to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# Calculate betas and alpha_bars once
betas, alpha_bars = noise_schedule(T)
    
# Diffusion Training Loop
loss = 0
for i in tqdm(range(len(dicom_files))):
    tensor = load_dicom_to_tensor(dicom_files[i])
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.randint(0, T, size = tensor.size())  # Random time step for each image

    # Forward diffusion
    x_t, noise = forward_diffusion(tensor, t)

    # Predict noise with the UNet model
    noise_pred = model(image=x_t, t=t)

    # Loss is between actual noise and predicted noise
    loss = mse_loss(noise_pred, noise)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss += loss.item()
    print("One run done.")
        
avg_loss = loss / len(dicom_files)
print(f"Average Loss: {avg_loss}")
print("Training complete.")

""" 
    # Save the best model based on lowest loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f"New best model saved with loss {best_loss:.6f}")

    # Save generated images after every epoch
    with torch.no_grad():
        sample_noise = torch.randn((64, 1, 28, 28)).to(x.device)
        for i in range(T - 1, 0, -1):
            sample_noise = reverse_diffusion_step(sample_noise, torch.tensor([i]).to(x.device), model, betas)
        save_image(sample_noise, f"./checkpoints/diff_generated_images_epoch_{epoch + 1}.png")

print("Training complete.")
"""
