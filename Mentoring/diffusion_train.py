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



# Model: UNet architecture (simplified)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, t):
        t_embed = torch.rand_like(x) * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_embed], dim=1)
        print(x.size())
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

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

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

img_dir = '/Users/hatapd/PycharmProjects/OCSEF/archive/head_ct/head_ct_train'
dataset = ImageDataset(img_dir)


model = UNet().to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# Calculate betas and alpha_bars once
betas, alpha_bars = noise_schedule(T)


for i in range(5):
    epoch_loss = 0
    image = dataset[i]
    results = transforms.Resize(256)(image)
    results = transforms.ToTensor()(results)

    results = results.to('cuda' if torch.cuda.is_available() else 'cpu')


    t = torch.randint(0, T, size=results.size())  # Random time step for each image
    x_t, noise = forward_diffusion(results, t)
    noise_pred = model(x_t, t)

    loss = mse_loss(noise_pred, noise)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()


"""
    
# Diffusion Training Loop
for epoch in range(epochs):
    print(f"Epoch [{epoch + 1}/{epochs}]")
    epoch_loss = 0.0

    image = Image.open(dataset_path)
    results = transforms.ToTensor()(image)
    print(results)

    
        #image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.randint(0, T, size = results.size())  # Random time step for each image

        # Forward diffusion
        x_t, noise = forward_diffusion(results, t)

        # Predict noise with the UNet model
        noise_pred = model(x_t, t)

        # Loss is between actual noise and predicted noise
        loss = mse_loss(noise_pred, noise)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataset_path)
    print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

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
