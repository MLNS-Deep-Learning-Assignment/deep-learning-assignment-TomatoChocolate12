import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

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

def train_model():
    # Parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dataset and train/val split
    full_dataset = DigitSumDataset(transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNNBaseline().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        training_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device).float(), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        validation_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, "best_digit_sum_model.pth")

    print("Training completed!")
    return model, training_losses, validation_losses

def infer_sum(model, image, transform, device):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device).float()
        output = model(image)
        predicted_sum = output.item()
        return predicted_sum

if __name__ == "__main__":
    model, train_losses, val_losses = train_model()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()
