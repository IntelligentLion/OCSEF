import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from networks.Encoder import Encoder
from networks.Decoder import Decoder
from networks.VAE import VAE

# Hyperparameters
dataset_path = '/PycharmProjects/OCSEF'
batch_size = 20
x_dim = 784  # MNIST images are 28x28, flattened to 784
hidden_dim = 400
latent_dim = 200
lr = 1e-3
epochs = 30
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# Data Transformations
mnist_transform = transforms.Compose([transforms.ToTensor()])

# Datasets and DataLoaders
train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model Setup
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(Encoder=encoder, Decoder=decoder).to(DEVICE)

# Optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Loss Function
def loss_function(x, x_hat, mean, log_var):
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return recon_loss + kl_div

# Function to validate model performance on test set
def validate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.view(x.size(0), x_dim).to(DEVICE)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

# Initialize variables to track the best model
best_loss = float('inf')
best_epoch = 0
best_model_path = "best_vae_model.pth"

# Training Loop
print("Start training VAE...")
for epoch in range(epochs):
    model.train()
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(x.size(0), x_dim).to(DEVICE)  # Reshape to [batch_size, 784]
        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = overall_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validate on test set
    avg_test_loss = validate_model(model, test_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Test Loss: {avg_test_loss:.4f}")

    # Save the model checkpoint if it has the best validation loss
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {best_epoch} with loss: {best_loss:.4f}")

    # Save reconstructed images every 10 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            sample = torch.randn(64, latent_dim).to(DEVICE)
            generated_images = model.decoder(sample).view(-1, 1, 28, 28)
            save_image(generated_images, f"reconstructed_epoch_{epoch + 1}.png")

print("Training complete!")
print(f"Best model was saved at epoch {best_epoch} with test loss: {best_loss:.4f}")
