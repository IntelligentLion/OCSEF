#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_node', type=str, help='specify gpu nodes')
parser.add_argument('--database', type=str, help='choose RadImageNet/ImageNet/CT')
parser.add_argument('--model_name', type=str, help='choose IRV2/ResNet50/DenseNet121/InceptionV3')
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--image_size', type=int, help='image size')
parser.add_argument('--structure', type=str, help='unfreezeall/freezeall/unfreezetop10', default='unfreezeall')
parser.add_argument('--epoch', type=int, help='number of epochs', default=30)

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_node}' if torch.cuda.is_available() else 'cpu')

### Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['dir']
        image = default_loader(img_path)
        label = int(self.dataframe.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

### Transformations
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(args.image_size)
])

### Load Data
df_train = pd.read_csv("OCSEF/dataset/hemorrhage/hemorrhage_train.csv")
df_val = pd.read_csv("OCSEF/dataset/hemorrhage/hemorrhage_val.csv")

train_dataset = CustomDataset(dataframe=df_train, transform=transform)
val_dataset = CustomDataset(dataframe=df_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

### Model Selection
def get_model():
    if args.model_name == 'IRV2':
        model = models.inception_v3(pretrained=True)
    elif args.model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif args.model_name == 'DenseNet121':
        model = models.densenet121(pretrained=True)
    elif args.model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=True)
    else:
        raise ValueError("Model name not supported")

    # Unfreeze layers based on structure argument
    if args.structure == 'freezeall':
        for param in model.parameters():
            param.requires_grad = False
    elif args.structure == 'unfreezetop10':
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False

    # Modify the final layer to match the number of classes
    if args.model_name in ['ResNet50', 'DenseNet121']:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
    elif args.model_name in ['IRV2', 'InceptionV3']:
        model.AuxLogits.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.AuxLogits.fc.in_features, 2)
        )
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 2)
        )
    return model

model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


### Training Loop
for epoch in range(args.epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/{args.epoch}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

### Save Model
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), f'models/hemorrhage_{args.structure}_{args.database}_{args.model_name}_{args.image_size}_{args.batch_size}_{args.lr}.pth')

### Save Training Loss
if not os.path.exists('loss'):
    os.makedirs('loss')
loss_data = {
    'epoch': list(range(1, args.epoch + 1)),
    'train_loss': running_loss,
    'val_loss': val_loss
}
df_loss = pd.DataFrame(loss_data)
df_loss.to_csv(f'loss/hemorrhage_{args.structure}_{args.database}_{args.model_name}_{args.image_size}_{args.batch_size}_{args.lr}.csv', index=False)