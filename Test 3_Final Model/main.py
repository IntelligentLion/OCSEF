import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
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
from Structure_init import UNetViTCross
from torchmetrics.image.fid import FrechetInceptionDistance




betas_total = torch.linspace(0.0001, 0.02, 1000) # all the betas within the specified range
alphas_total = 1. - betas_total # all the alphas within the range

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

#Hyperparameters
epochs = 1
batch_size = 2
lr = 2e-4
T = 5 # Number of timesteps in the forward diffusion process
best_loss = float('inf')  # Initialize best loss with infinity


cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


model = UNetViTCross()
model = model.to(DEVICE)

betas, alpha_bars = noise_schedule(T)

optimizer = optim.Adam(model.parameters(), lr=lr)
fid = FrechetInceptionDistance(normalize=True)
mse_loss = nn.MSELoss()

loss = 0
for i in tqdm(range(len(dicom_files))):
    tensor = load_dicom_to_tensor(dicom_files[i])
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    fid.update(tensor, real = True)
    t = torch.randint(0, T, size=tensor.size())  # Random time step for each image

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


    # Initialize your model, optimizer, and loss function
    model = UNetViTCross()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    fid = FrechetInceptionDistance(feature=64)

    # Define your training and evaluation parameters
    num_epochs = 100  # Total number of epochs for training
    eval_interval = 5  # Perform validation every 5 epochs
    fid_weight = 0.3  # Weight for the FiD score in the combined loss
    fid_scores = []  # List to store FiD scores

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for batch in tqdm(train_dataloader):  # Iterate over batches
            # Forward pass
            outputs, timestep_embeddings = model(batch['input'])

            # Compute MSE for timestep embeddings
            mse = mse_loss(timestep_embeddings, batch['timestep_embeddings'])

            # Compute FiD score periodically
            if epoch % eval_interval == 0:
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    generated_images = model.generate_images(batch_size=64)
                    fid.update(generated_images, real=False)
                    fid.update(val_batch['real_images'], real=True)
                fid_score = fid.compute().item()
                fid.reset()
                fid_scores.append(fid_score)

                # Update combined loss
                combined_loss = mse + fid_weight * fid_score
            else:
                combined_loss = mse

            # Backward pass and optimization
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

        # Validation loop for FiD score
        if epoch % eval_interval == 0:
            print(f'Epoch {epoch}, Combined Loss: {combined_loss.item()}, Validation FiD: {fid_score}')

    # Print final FiD scores
    print('Final FiD Scores:', fid_scores)

    loss += loss.item()
    print("One run done.")

avg_loss = loss / len(dicom_files)
print(f"Average Loss: {avg_loss}")
print("Training complete.")