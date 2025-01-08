import numpy as np
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset


class DigitSumDataset(Dataset):
    def __init__(self, data_files_pattern='../data/data*.npy', label_files_pattern='../data/lab*.npy', transform=None):
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
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 40, 168)
            dummy_output = self.conv_layers(dummy_input)
            flatten_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x