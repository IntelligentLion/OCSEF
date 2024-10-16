import torch
from torch import nn
from torch import optim
import torch.nn.functional as F #Non-parametrized functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from medmnist import OrganCMNIST

#Iniitializing Data
Train_Data = OrganCMNIST(split = 'train', download = True, size = 64)
Val_Data = OrganCMNIST(split = 'val', download = True, size = 64)
Test_Data = OrganCMNIST(split = 'test', download = True, size = 64)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.convL1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.convL2 = nn.LazyConv2d(out_channels = 16, kernel_size = 5, stride = 1, padding = 1)
        self.linL3 = nn.LazyLinear(out_features=24, bias = True)
        self.poolL4 = nn.MaxPool2d

