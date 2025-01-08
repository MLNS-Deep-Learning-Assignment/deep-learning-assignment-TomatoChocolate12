import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class DigitSumDataset(Dataset):
    def __init__(self, data_files_pattern='test_data*.npy', label_files_pattern='test_lab*.npy', transform=None):
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

def evaluate_test_set(model_path, test_data_pattern='test_data*.npy', test_label_pattern='test_lab*.npy'):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = CNNBaseline().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print training metrics from checkpoint
    print("\nModel checkpoint info:")
    print(f"Saved at epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    if 'metrics' in checkpoint:
        print("Best validation metrics:")
        for k, v in checkpoint['metrics'].items():
            print(f"{k}: {v:.4f}")
    print("-" * 50)

    # Data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load test dataset
    test_dataset = DigitSumDataset(
        data_files_pattern=test_data_pattern,
        label_files_pattern=test_label_pattern,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluation metrics
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.MSELoss()

    # Evaluate
    print("\nEvaluating test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device).float(), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to tensors for easier computation
    predictions = torch.tensor(all_predictions)
    true_labels = torch.tensor(all_labels)

    # Calculate metrics
    floor_pred = torch.floor(predictions.squeeze())
    ceil_pred = torch.ceil(predictions.squeeze())
    round_pred = torch.round(predictions.squeeze())

    floor_accuracy = (floor_pred == true_labels).float().mean().item()
    ceil_accuracy = (ceil_pred == true_labels).float().mean().item()
    round_accuracy = (round_pred == true_labels).float().mean().item()
    mae = torch.abs(predictions.squeeze() - true_labels).mean().item()
    mse = ((predictions.squeeze() - true_labels) ** 2).mean().item()

    # Print results
    print("\nTest Set Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Floor Accuracy: {floor_accuracy:.4f}")
    print(f"Ceiling Accuracy: {ceil_accuracy:.4f}")
    print(f"Rounded Accuracy: {round_accuracy:.4f}")

    # Plot prediction distribution
    plt.figure(figsize=(15, 5))
    
    # Prediction vs True Value scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(true_labels.numpy(), predictions.squeeze().numpy(), alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')

    # Error distribution
    plt.subplot(1, 2, 2)
    errors = predictions.squeeze().numpy() - true_labels.numpy()
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

    return {
        'mse': mse,
        'mae': mae,
        'floor_accuracy': floor_accuracy,
        'ceil_accuracy': ceil_accuracy,
        'round_accuracy': round_accuracy,
        'predictions': predictions.numpy(),
        'true_labels': true_labels.numpy()
    }

if __name__ == "__main__":
    # Usage example
    results = evaluate_test_set(
        model_path='best_digit_sum_model.pth',
        test_data_pattern='../data/data*.npy',
        test_label_pattern='../data/lab*.npy'
    )