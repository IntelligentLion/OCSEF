import os
from glob import glob
import numpy as np
import torch
import pydicom

data_dir = '/Users/humza/PycharmProjects/OCSEF/dataset/ICH_train'
dicom_files = sorted(glob(os.path.join(data_dir, "*.dcm")))
def load_dicom_to_tensor(file_path):
    # Load DICOM file
    dicom = pydicom.dcmread(file_path)

    # Print some basic metadata to verify correct reading
    print(f"Processing file: {file_path}")
    print(f"Patient ID: {dicom.PatientID if 'PatientID' in dicom else 'N/A'}")
    print(f"Modality: {dicom.Modality if 'Modality' in dicom else 'N/A'}")

    # Get pixel array and convert to float32
    image = dicom.pixel_array.astype(np.float32)

    # Ensure 2D image is in (H,W) format
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]  # Add channel dimension: (1,H,W)
    elif len(image.shape) == 3:
        image = image[0:1, ...]  # Take first channel if multi-channel

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(np.array(image).astype('float32'))
    return tensor

tensors = []
for file_path in dicom_files:
    tensor = load_dicom_to_tensor(file_path)
    tensors.append(tensor)

batch = torch.stack(tensors)  # Shape: [batch_size, 1, H, W]
print(f"Batch shape: {batch.shape}")
batch = (batch - batch.min()) / (batch.max() - batch.min())
print(f"Normalized range: [{batch.min():.3f}, {batch.max():.3f}]")
