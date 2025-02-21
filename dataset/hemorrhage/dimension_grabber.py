import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class DICOMDataset(Dataset):
    def __init__(self, dicom_paths, transform=None):
        self.dicom_paths = dicom_paths
        self.transform = transform

    def __len__(self):
        return len(self.dicom_paths)

    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array

        if self.transform:
            image = self.transform(image)

        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        return image_tensor, 0  # Returns image and a dummy label (0)

dicom_dir = '/Users/humza/PycharmProjects/OCSEF/dataset/ICH_train'
dicom_paths = [os.path.join(dicom_dir, filename) for filename in os.listdir(dicom_dir) if filename.endswith('.dcm')]

dataset = DICOMDataset(dicom_paths)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_idx, images in enumerate(dataloader):
    print(images)

