# Imports

# torch.nn
import torch
from torch import nn as nn
from sympy.physics.units import acceleration
from torch import optim
import torch.nn.functional as F #Non-parametrized functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network

class NN(nn.Module):
    def __init__(self, input_size, num_classes): # (28x28 = 784)
        super(NN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes) #dataset specific features

    def forward(self, y):
        x = F.relu(self.fc1(y)) #forward for forward prop by calling functions per layer
        x = self.fc2(x)
        return x

#model = NN(784, 10)
#x = torch.rand(64,784)
# #print(model(x).shape)  Test from video for NN module

#test_nn = NN(54, 24)
#print(test_nn.input_size)
#print(test_nn.num_classes)

#Set Device
#   Checking for CUDA capable device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes = num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train Network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #Get data to CUDA if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #flatten vector to desired shape
        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        print(scores)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad() #reset gradients to zero
        loss.backward() #backpropagation

        #gradient descent
        optimizer.step()
#Check accuracy on training and test to see model quality

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval() #setting evaluation mode on the model

    with torch.no_grad(): #avoiding gradient computation
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100.2}")

model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
