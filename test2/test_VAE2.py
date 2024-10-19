import torch
from torch import nn
from torch import optim
import torch.nn.functional as F #Non-parametrized functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class VAE(nn.Module):
    batch_size = 20
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()

        # input_dim = 28*28

        #encoder
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 7, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #latent space -- mean and log variance
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_log_var = nn.Linear(h_dim, z_dim)

        #fully connected decoding layer ---
        self.decode = nn.Linear(z_dim, h_dim)

        #Decoder Layers
        self.deconv1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1 )
        self.deconv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 5, kernel_size = 5, stride = 1, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = 7, stride = 2, padding = 1)

    def encode(self, img_in):
        img_in = F.gelu(self.conv1(img_in))
        img_in = F.gelu(self.conv2(img_in))
        img_in = F.gelu(self.conv3(img_in))
        img_in = self.pool4(img_in)
        img_in = img_in.view(img_in.size(0), -1)
        mu = self.fc_mu(img_in)
        log_var = self.fc_log_var(img_in)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 28, 28, 1)
        z = F.gelu(self.deconv1(z))
        z = torch.gaussian_nll_loss(self.deconv2(z))

        return z

    def forward(self, img_in):
        mu, log_var = self.encode(img_in)
        z1 = self.reparameterize(mu, log_var)
        return self.decode(z1), mu, log_var

'''if __name__ == "__main__":
    x = torch.randn(4, 28*28, 1) # 28*28 = 784
    vae = VAE(input_dim=784)
    z1, mu, log_var = vae(x)
    print(z1.shape)
    print(mu.shape)
    print(log_var.shape)
'''

