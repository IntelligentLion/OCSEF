# Imports

# torch.nn
import torch
from torch import nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network

class NN:
    def __init__(self, input_size, num_classes): # (28x28 = 784)
        #super(NN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        #self.fc1 = nn.Linear(input_size, 50)
        #self.fc2 = nn.Linear(50, num_classes)

    #def forward(self, y):
        #x = F.relu(self.fc1(y))
        #x = self.fc2(x)
        #return x

test_nn = NN(54, 24)
print(test_nn.input_size)