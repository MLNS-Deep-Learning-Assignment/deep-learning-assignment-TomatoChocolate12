import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class DigitSumDataset(Dataset):
    def __init__(self, data_files_pattern='data*.npy', label_files_pattern='lab*.npy', transform=None):
        self.data_files = sorted(glob.glob(data_files_pattern))
        self.label_files = sorted(glob.glob(label_files_pattern))
        self.transform = transform
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            data.append(np.load(data_file))
            labels.append(np.load(label_file))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

class CNNBaseline(nn.Module):
    def __init__(self):  
        super(CNNBaseline, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 1 channel for grayscale
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Update the fully connected layer to expect 26880 inputs after flattening
        self.fc_layers = nn.Sequential(
            nn.Linear(26880, 128),  # Adjusted to the correct flattened size
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value for the sum
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x
