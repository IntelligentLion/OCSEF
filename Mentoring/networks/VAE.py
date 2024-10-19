import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Convert log variance to standard deviation
        epsilon = torch.randn_like(std).to(std.device)  # Random noise
        z = mean + std * epsilon  # Reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
