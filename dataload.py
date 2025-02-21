import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torchvision.transforms as transforms
from PIL import Image

image = Image.open('/Users/hatapd/PycharmProjects/OCSEF/archive/head_ct/head_ct_train/001.png')
results = transforms.ToTensor()(image)
print(results)



class CustomDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('/Users/hatapd/PycharmProjects/OCSEF/archive/labels.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = CustomDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
total_samples = len(dataset)
num_iterations = math.ceil(total_samples/4)
print(total_samples, num_iterations)
