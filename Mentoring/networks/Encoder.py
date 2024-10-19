import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leaky_relu(self.fc_input(x))
        h = self.leaky_relu(self.fc_hidden(h))
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var
