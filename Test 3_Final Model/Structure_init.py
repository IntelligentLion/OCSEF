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
from ViT import VisionTransformer, PatchEmbed, DualInputCrossAttention, Block

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
    x = x.view(B, C, H * W)  # Reshape to (H*W, B, C)

        # Extract attention features for the current stage
    attn_features = attention_map[stage-1, :, :]  # Shape: (256, 784)

        # Perform matrix multiplication with attention features
    weighted_x = torch.matmul(attn_features, x)

    weighted_x = weighted_x.view(B, C, H, W)

    return weighted_x

# Model: UNet architecture (simplified)
class UNetViTCross(nn.Module):
    def __init__(self):
        super(UNetViTCross, self).__init__()

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

        self.cross_model = VisionTransformer()

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

        x9 = self.doubleConv_5(x8) # bottleneck, Shape: [1, 1024, 28, 28]

        x_reshaped = x9.flatten(2).transpose(1, 2)
        print(x_reshaped.shape)

        x_i = self.cross_model(feature_tensor, x_reshaped)
        print(x_i.shape)

        # decoder and feature addition
        #x9 = feature_addition(x9, x_i, 1)
        x_up_1 = self.up_trans_1(x9)
        y_1 = crop_img(x7, x_up_1)
        #y_1 = self.cross_model(y_1) # Apply cross attention
        z1 = torch.cat([x_up_1, y_1], 1)
        x_cat_1 = self.up_conv_1(z1)

       #x_cat_1 = feature_addition(x_cat_1, x_i, 2)
        x_up_2 = self.up_trans_2(x_cat_1)
        y_2 = crop_img(x5, x_up_2)
        z2 = torch.cat([x_up_2, y_2], 1)
        x_cat_2 = self.up_conv_2(z2)  ######################

        #x_cat_2 = feature_addition(x_cat_2, x_i, 3)
        x_up_3 = self.up_trans_3(x_cat_2)
        y_3 = crop_img(x3, x_up_3)
        z3 = torch.cat([x_up_3, y_3], 1)
        x_cat_3 = self.up_conv_3(z3)

        #x_cat_3 = feature_addition(x_cat_3, x_i, 4)
        x_up_4 = self.up_trans_4(x_cat_3)
        y_4 = crop_img(x1, x_up_4)
        z4 = torch.cat([x_up_4, y_4], 1)
        x_cat_4 = self.up_conv_4(z4)

        output = self.out(x_cat_4)
        print(output.shape)
        return output

data_dir = ('/Users/hatapd/PycharmProjects/OCSEF/dataset/ICH_train')
dicom_files = sorted(glob(os.path.join(data_dir, "*.dcm")))

def load_dicom_to_tensor(file_path):
    # Load DICOM file
    dicom = pydicom.dcmread(file_path, force=True)

    # Print some basic metadata to verify correct reading
    #print(f"Processing file: {file_path}")
    #print(f"Patient ID: {dicom.PatientID if 'PatientID' in dicom else 'N/A'}")
    #print(f"Modality: {dicom.Modality if 'Modality' in dicom else 'N/A'}")

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


model = UNetViTCross().to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

loss = 0
for i in tqdm(range(len(dicom_files)-4)):
    tensor = load_dicom_to_tensor(dicom_files[i])
    tensor = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    output = model(tensor)
    print(output.shape)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss += loss.item()
    print("One run done.")

#test = nn.Conv2d(1, 1024, kernel_size=1)
#x = test(x)
