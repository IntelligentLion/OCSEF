import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, z):
        h = self.leaky_relu(self.fc_hidden(z))
        h = self.leaky_relu(self.fc_hidden2(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat
