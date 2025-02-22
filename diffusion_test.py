import torch
import torch.nn as nn
from math import lgamma
from fontTools.misc.bezierTools import epsilon
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datafmt="%I:%M:%S")

#hyperparameters
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

class Diffusion:
    def __init(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device=DEVICE):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def forward_process(self,x_0 , t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] #I don't know what the "[:, None, None, None]" part is
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon_val = torch.randn_like(x_0)
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon_val, epsilon_val

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) # noised image
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device) # random timestep
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1/torch.sqrt(alpha) * (x - ((1-alpha)/(torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1,1) + 1) / 2 # could replace the clamp() function with a diff (-1, 1) limiting function -- i think it was sigmoid, but there should be some function that we can use
        x = (x*255).type(torch.uint8)
        return x








    






