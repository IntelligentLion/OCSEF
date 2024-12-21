import torch
import torchvision.datasets as datasets # Standard Datasets
from tqdm import tqdm
from torch import nn

from Mentoring.diffusion_train import mse_loss
from test_VAE2 import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader # Gives easier data management by creating mini batches, etc.
import medmnist
from medmnist import OrganAMNIST
from test_nn import train_loader, batch_size

#Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 640
Z_DIM = 20
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR_RATE = 1e-3 #Karapathy constant

#datasets
train_dataset = OrganAMNIST(split="train", download=True)
val_dataset = OrganAMNIST(split="val", download=True)
test_dataset = OrganAMNIST(split="test", download = True)

#loaders
train_loader = DataLoader(dataset= train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset = val_dataset, batch_size=BATCH_SIZE, shuffle =False)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

#Model and Parameters
model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss(reduction = 'mean') #Other options, MSE


#Forward Training
def train(model, train_loader, val_loader, epochs, optimizer, loss, device):
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(enumerate(train_loader))
        train_loss = 0.0
        for i, (x, _) in loop:
            #pass to CUDA
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, log_var = model(x)

            #compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            #backprop/Parameterization trick
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            train_loss += loss.item()

        #validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                x_reconstructed, mu, log_var = model(x)
                reconstruction_loss = mse_loss(x_reconstructed, x)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + kl_loss
                val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {val_loss:.6f}")
            return val_loss

#train the model
val_loss = train(model, train_loader, val_loader, NUM_EPOCHS, optimizer, mse_loss, DEVICE)

#test loop

def test(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_reconstructed, mu, log_var = model(x)
            recon_loss = mse_loss(x_reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) + log_var.exp())
            loss = recon_loss + kl_loss
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss :.6f}")
    return test_loss

test_loss = test(model, test_loader, DEVICE)

#checkpoints -- from Mentoring\train.py
best_loss = float('inf')
best_epoch = 0
best_model_path = "best_convae_path" # Save the model checkpoint if it has the best validation loss

if test_loss < best_loss:
    best_loss = val_loss
    best_epoch = epoch + 1
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved at epoch {best_epoch} with loss: {best_loss:.4f}")

    # Save reconstructed images every 5 epochs
if (epoch + 1) % 5 == 0:
    with torch.no_grad():
        sample = torch.randn(64, latent_dim).to(DEVICE)
        generated_images = model.decoder(sample).view(-1, 1, 28, 28)
        save_image(generated_images, f"reconstructed_epoch_{epoch + 1}.png")

print("Training complete!")
print(f"Best model was saved at epoch {best_epoch} with test loss: {best_loss:.4f}")


'''
model = model.to("cpu")
def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append()
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
          mu, log_var = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, log_var = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + log_var * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=1)'''