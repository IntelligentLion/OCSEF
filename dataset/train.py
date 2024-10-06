import torch
import torchvision.datasets as datasets # Standard Datasets
from tqdm import tqdm
from torch import nn
from test_VAE1 import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader # Gives easier data management by creating mini batches, etc.