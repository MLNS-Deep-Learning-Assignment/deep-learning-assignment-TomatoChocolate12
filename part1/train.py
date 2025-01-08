import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

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

def calculate_accuracy(outputs, labels):
    # Calculate accuracy using floor
    floor_pred = torch.floor(outputs.squeeze())
    floor_accuracy = (floor_pred == labels).float().mean().item()
    
    # Calculate accuracy using ceiling
    ceil_pred = torch.ceil(outputs.squeeze())
    ceil_accuracy = (ceil_pred == labels).float().mean().item()
    
    # Calculate accuracy taking the closest integer
    round_pred = torch.round(outputs.squeeze())
    round_accuracy = (round_pred == labels).float().mean().item()
    
    # Calculate mean absolute error
    mae = torch.abs(outputs.squeeze() - labels).mean().item()
    
    return {
        'floor_accuracy': floor_accuracy,
        'ceil_accuracy': ceil_accuracy,
        'round_accuracy': round_accuracy,
        'mae': mae
    }

def train_model():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

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

    best_val_loss = float('inf')
    no_improve = 0
    training_losses = []
    validation_losses = []
    validation_metrics = []

    for epoch in range(num_epochs):
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

        model.eval()
        running_val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device).float(), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                running_val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        validation_losses.append(val_loss)

        val_predictions = torch.tensor(val_predictions)
        val_true_labels = torch.tensor(val_true_labels)
        metrics = calculate_accuracy(val_predictions, val_true_labels)
        validation_metrics.append(metrics)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Floor Accuracy: {metrics['floor_accuracy']:.4f}")
        print(f"Ceiling Accuracy: {metrics['ceil_accuracy']:.4f}")
        print(f"Rounded Accuracy: {metrics['round_accuracy']:.4f}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print("-" * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics
            }, "best_digit_sum_model.pth")


    print("Training completed!")
    return model, training_losses, validation_losses, validation_metrics

def infer_sum(model, image, transform, device):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device).float()
        output = model(image)
        predicted_sum = output.item()
        return {
            'raw_prediction': predicted_sum,
            'floor_prediction': int(np.floor(predicted_sum)),
            'ceil_prediction': int(np.ceil(predicted_sum)),
            'round_prediction': int(np.round(predicted_sum))
        }

if __name__ == "__main__":
    model, train_losses, val_losses, val_metrics = train_model()
    
    # Plot training and validation losses
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    epochs = range(len(val_metrics))
    plt.plot(epochs, [m['floor_accuracy'] for m in val_metrics], label='Floor Accuracy')
    plt.plot(epochs, [m['ceil_accuracy'] for m in val_metrics], label='Ceiling Accuracy')
    plt.plot(epochs, [m['round_accuracy'] for m in val_metrics], label='Rounded Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracies')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()