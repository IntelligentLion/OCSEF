import torch
from torch import nn
from sympy.physics.units import acceleration
from torch import optim
import torch.nn.functional as F #Non-parametrized functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
       super().__init__()

    def encode(self, x):
        # q_phi(
        pass

