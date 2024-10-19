import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from networks.Encoder import Encoder
from networks.Decoder import Decoder
from networks.VAE import VAE

# Hyperparameters and Configurations
dataset_path = 'C:/Users/himas/Documents/Mentoring/datasets'
batch_size = 10
x_dim = 784  # MNIST images are 28x28, flattened to 784
hidden_dim = 400
latent_dim = 200
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# Data Transformations
mnist_transform = transforms.Compose([transforms.ToTensor()])

# Datasets and DataLoaders
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(Encoder=encoder, Decoder=decoder).to(DEVICE)

# Load trained weights (make sure to specify the path to your saved model)
checkpoint = torch.load('best_vae_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint)

# Evaluation mode
model.eval()

def show_image(x, idx):
    """Visualize the image."""
    x = x.view(batch_size, 28, 28)
    plt.imshow(x[idx].cpu().numpy(), cmap='gray')
    plt.title(f"Image {idx + 1}")
    plt.axis('off')
    plt.show()

# Run inference on the test dataset
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        # Flatten the input image to [batch_size, 784]
        x = x.view(batch_size, x_dim).to(DEVICE)

        # Forward pass through the model
        x_hat, _, _ = model(x)

        # Reshape x and x_hat to visualize the original and reconstructed images
        x = x.view(batch_size, 1, 28, 28)
        x_hat = x_hat.view(batch_size, 1, 28, 28)

        # Show original image
        print(f"Original Image - Batch {batch_idx + 1}")
        show_image(x, idx=0)  # Display the first image in the batch

        # Show reconstructed image
        print(f"Reconstructed Image - Batch {batch_idx + 1}")
        show_image(x_hat, idx=0)  # Display the first reconstructed image

        # Optionally break after one batch to avoid visualizing all batches
        if batch_idx == 0:  # Comment this out if you want to evaluate all batches
            break
