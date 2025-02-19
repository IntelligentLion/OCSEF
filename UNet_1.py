import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
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



## NEED TO CHANGE CHANNEL SIZE OF INPUT TO 2 channels, AS IT SHOULD BE 2 NOT 6!

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

        print(output.size())


# Betas and alphas
timesteps = 200
betas_total = torch.linspace(0.01, 0.5, timesteps)
alphas_total = 1 - betas_total

def noise_schedule(t):
    beta_t = betas_total[t-1]
    alpha_t = alphas_total[t-1]

    alpha_t_bar = torch.cumprod(alphas_total, dim=0)
    return beta_t, alpha_t_bar

def forward_diffusion(x_0, timestep):
    alpha_t_bar = alphas_total[timestep-1]
    noise = torch.randn(size = x_0.size())

    alpha_t_bar = alpha_t_bar.view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1-alpha_t_bar) * noise

    return x_t, noise

def reverse_diffusion(x_t, model, t, betas): # Watch the video explaining the reverse diffusion process!!!!
    beta_t, alpha_t_bar = noise_schedule(t)
    noise_pred = model(x_t, t)

    return (x_t - beta_t * noise_pred) / (torch.sqrt(alpha_t_bar))

# Hyperparameters
EPOCHS = 10
lr = 1e-4 # karapathy constant
BATCH_SIZE = 12
best_loss = float('inf')

# Model Checkpoint Path
checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

dataset_path = '/Users/hatapd/PycharmProjects/OCSEF/archive/head_ct/head_ct_train'
labels = '/Users/hatapd/PycharmProjects/OCSEF/archive/head_ct/labels.csv'
transform = transforms.Compose([transforms.ToTensor()])

Dataset = CustomDataset(annotations_file=labels, img_dir=dataset_path, transform=transform)

dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Initialize model
model = UNet().to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# Calculate the betas and alpha_bars once
betas, alpha_bars = noise_schedule(timesteps)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}/{EPOCHS}")
    epoch_loss = 0
    for batch_idx, (image, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        print(image.size())

        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.randint(0, timesteps, (1,)).to(image.device)

        # Forward Diffusion
        x_t, noise = forward_diffusion(image, t)

        # Predict noise
        noise_pred = model(x_t, t)

        #Loss
        loss = mse_loss(noise_pred, noise)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # Save the best model based on lowest loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f"New best model saved wit loss {best_loss:.6f}")

    # Save generated images after every epoch
    with torch.no_grad():
        sample_noise = torch.randn((64, 1, 28, 28)).to(image.device)
        for i in range(timesteps - 1, 0, -1):
            sample_noise = reverse_diffusion(sample_noise, torch.tensor([i]).to(image.device), model, betas)
        save_image(sample_noise, f"./checkpoints/diff_generated_images_epeoch_{epoch+1}.png")

print("Training complete.")
