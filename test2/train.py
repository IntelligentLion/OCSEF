import torch
import torchvision.datasets as datasets # Standard Datasets
from tqdm import tqdm
from torch import nn
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
LR_RATE = 1e-4 #Karapathy constant

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
loss_fn = nn.GaussianNLLLoss(reduction = 'mean') #Other options, MSE


#Forward Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        #pass to CUDA
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, log_var = model(x)

        #compute loss
        reconstruction_loss = loss_fn(torch.exp(log_var))
        kl_div = torch.sum(1+torch.log(log_var.pow(2) - mu.pow(2) - log_var.pow(2)))

        #backprop/Parameterization trick
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

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
    inference(idx, num_examples=1)