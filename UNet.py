import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.utils import save_image
from torch import dataset
from dataset import MNIST
import tqdm
import pandas as pd

def crop_img(tensor, target_tensor):
    tensor_size = tensor.size()[2]
    target_size = target_tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

def doubleConv2d(in_c, out_c):
    conv = nn.Sequential(?
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


    def forward(self, image):
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
        x_cat_2 = self.up_conv_2(z2)

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
    noise = torch.randn_like(x_0)

    alpha_t_bar = alpha_t_bar.view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_t_bar) * x_0 + torch.sqrt(1-alpha_t_bar) * noise

    return x_t, noise

def reverse_diffusion(x_t, model, t, betas): # Watch the video explaining the reverse diffusion process!!!!
    beta_t, alpha_t_bar = noise_schedule(t)
    noise_pred = model(x_t, t)

    return (x_t - beta_t * noise_pred) / (torch.sqrt(alpha_t_bar))

# Hyperparameters
EPOCHS = 10
lr = 1e-4 #karapathy constant
BATCH_SIZE = 12
best_loss = float('inf')

# Dataset and Dataloader
dataset_path = "~/OCSEF/dataset/hemorrhage/hemorrhage_train.csv"
data_Transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root=dataset_path, train=True, download=True, transform=data_Transform)

dataset_size = len(train_dataset)
train_size = int(0.5 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}/{EPOCHS}")
    epoch_loss = 0
    for batch_idx, (image, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        t = torch.randint(0, timesteps, (1,))

        # Forward Diffusion


if __name__ == "__main__":
    betas = beta_schedule(17, 0, 0.05)
    random_timestep = get_random_timestep(betas, 0, len(betas)-1)
    print(random_timestep)