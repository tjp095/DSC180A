#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import math
import random


# In[6]:


class RotatedMNIST(Dataset):
    def __init__(self, base_dataset, min_digit: int, max_digit: int, min_rotation_deg: float, max_rotation_deg: float):
        self.base_dataset = base_dataset # the original MNIST dataset
        self.min_rotation_deg = min_rotation_deg
        self.max_rotation_deg = max_rotation_deg

        # Filter dataset using max_digit:
        valid_indices = torch.where((base_dataset.targets >= min_digit) & (base_dataset.targets <= max_digit))[0]
        self.indices = valid_indices.tolist()

    def __len__(self):
        # Return the length of the filtered dataset, not the whole dataset:
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the requested index to the actual index in the dataset:
        real_idx = self.indices[idx]

        # Retrieve the base image tensor and digit label:
        img_tensor, digit_label = self.base_dataset[real_idx]

        # Generate a random angle (0 to max_rotation parameter value)
        angle_deg = random.uniform(self.min_rotation_deg, self.max_rotation_deg)
        angle_rad = math.radians(angle_deg)

        # Rotate the image tensor
        img_rotated = TF.rotate(img_tensor, angle_deg, fill=0)

        # Convert continuous radian angle into a 2-value tensor (Sin and Cos)
        # $$$ Not using currently, but may consider if it improves accuracy
        # sin_cos_target = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)

        return img_rotated, digit_label, angle_rad, angle_deg


# In[3]:


def create_dataloaders(data_cfg: dict):
    '''
    Takes the 'data' section of the config file and returns PyTorch DataLoaders
    '''
    transform = transforms.ToTensor()

    base_train = datasets.MNIST(
        root = data_cfg['dataset_path'],
        train = True,
        download = True,
        transform = transform
    )

    base_test = datasets.MNIST(
        root = data_cfg['dataset_path'],
        train = False,
        download = True,
        transform = transform
    )

    # Wrap with custom Rotated dataset:
    train_data = RotatedMNIST(
        base_dataset=base_train, 
        min_digit=data_cfg.get('min_digit', 0), # Defaults to 0 if not found
        max_digit=data_cfg.get('max_digit', 9), # Defaults to 9 if not found
        min_rotation_deg=data_cfg.get('min_rotation_degrees', 0),
        max_rotation_deg=data_cfg['max_rotation_degrees']
    )
    
    test_data = RotatedMNIST(
        base_dataset=base_test, 
        min_digit=data_cfg.get('min_digit', 0),
        max_digit=data_cfg.get('max_digit', 9),
        min_rotation_deg=data_cfg.get('min_rotation_degrees', 0),
        max_rotation_deg=data_cfg['max_rotation_degrees']
    )

    # Using batch size and num workers from config:
    train_loader = DataLoader(
        train_data,
        batch_size = data_cfg['batch_size'],
        num_workers = data_cfg['num_workers'],
        shuffle = True
    )

    test_loader = DataLoader(
        test_data,
        batch_size = data_cfg['batch_size'],
        num_workers = data_cfg['num_workers'],
        shuffle = False
    )

    return train_loader, test_loader


# In[5]:


# ========================================================
#                    LOCAL SANITY CHECK
# ========================================================
if __name__ == "__main__":
    # 1. Create a mock config to test the pipeline locally
    mock_data_cfg = {
        'dataset_path': './DNN_Data',
        'batch_size': 32,
        'num_workers': 0,  # Set to 0 to avoid multiprocessing issues during plotting
        'max_digit': 9,
        'max_rotation_degrees': 360
    }

    print("Running Data Setup Sanity Check...")

    # 2. Get the dataloaders
    train_loader, _ = create_dataloaders(mock_data_cfg)

    # 3. Grab a single batch from the training loader
    images, digits, angle_rad, angle_deg = next(iter(train_loader))

    # 4. Set up the matplotlib figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for i in range(5):
        # Extract the image and remove the channel dimension [1, 28, 28] -> [28, 28]
        img = images[i].squeeze().numpy()

        # Extract values for this particular image
        digit = digits[i].item()
        angle = angle_deg[i].item()

        # Plot the image and set the title
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Digit: {digit}\nAngle: {angle:.0f}°")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:




