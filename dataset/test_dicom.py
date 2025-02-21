import pydicom
from pydicom import dcmread
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class DICOMDataset(Dataset):
    def __init__(self, dicom_dir, transform=None):
        self.dicom_dir = dicom_dir
        self.dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        self.transform = transform

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_path = os.path.join(self.dicom_dir, self.dicom_files[idx])
        dicom_image = pydicom.dcmread(dicom_path).pixel_array

        if self.transform:
            dicom_image = self.transform(dicom_image)

        #image_tensor = torch.from_numpy(dicom_image).float().unsqueeze(0)
        return dicom_image, 0  # Returns image and a dummy label (0)


path = '/Users/humza/PycharmProjects/OCSEF/dataset/ICH_train/ID_000a8ee70.dcm'
ds = pydicom.dcmread(path)
data = ds.pixel_array
tensor_value = torch.from_numpy(data)
print(tensor_value.shape)
