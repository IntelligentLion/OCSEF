import torch
import torch.nn as nn
from math import  lgamma

from fontTools.misc.bezierTools import epsilon
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

import diffusion_test
from diffusion_test import Diffusion

#hyperparameters
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class Unet(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, device = DEVICE): # for black and white images it should be 1 for the input and output, along with text, it would be 2
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down()

