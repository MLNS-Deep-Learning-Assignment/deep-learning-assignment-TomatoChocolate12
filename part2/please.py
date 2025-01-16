import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os


class DigitSegmentationDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])  # Load the image
        image = Image.fromarray(image)  # Convert to PIL image
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.image_files[idx]  # Return image and filename


def visualize_predictions(image, predictions):
    """Visualize the original image with predicted masks overlaid."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score > 0.5:  # Display high-confidence predictions
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                              fill=False, edgecolor='red', linewidth=2))
    plt.show()


def save_cropped_digits(image, predictions, output_dir):
    """Save cropped images of each digit."""
    for i, box in enumerate(predictions["boxes"]):
        x1, y1, x2, y2 = box.int().tolist()
        cropped = image[:, y1:y2, x1:x2]
        save_path = os.path.join(output_dir, f"digit_{i}.png")
        torchvision.utils.save_image(cropped, save_path)


# Load the dataset
data_files = sorted(glob.glob('../data/data0.npy'))
dataset = DigitSegmentationDataset(data_files, transform=torchvision.transforms.ToTensor())

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load a pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

output_dir = "segmented_digits"
os.makedirs(output_dir, exist_ok=True)

# Run inference
for images, filenames in dataloader:
    images = [img.to(device) for img in images]
    
    # Perform inference
    with torch.no_grad():
        predictions = model(images)
    
    for image, prediction, filename in zip(images, predictions, filenames):
        print(f"Processing: {filename}")
        visualize_predictions(image, prediction)
        save_cropped_digits(image, prediction, output_dir)
