import torchvision
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess the image
img = Image.open("../../../../scratch/yashas.b/images/data0_324.jpg").convert("RGB")
transform = torchvision.transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

# Convert image for display
img_array = np.array(img)

# Create figure and axis
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img_array)

# Get prediction results
boxes = predictions[0]['boxes'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()
masks = predictions[0]['masks'].cpu().numpy()

# Filter predictions with high confidence (e.g., > 0.5)
confidence_threshold = 0.5
high_conf_idx = scores > confidence_threshold

boxes = boxes[high_conf_idx]
scores = scores[high_conf_idx]
masks = masks[high_conf_idx]

# Draw bounding boxes and masks
for box, score, mask in zip(boxes, scores, masks):
    # Draw bounding box
    x1, y1, x2, y2 = box.astype(int)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    
    # Add score text
    ax.text(x1, y1-10, f'Score: {score:.2f}', color='red', fontsize=8, backgroundcolor='white')
    
    # Overlay mask
    mask = mask.squeeze()
    mask = mask > 0.5  # Convert to binary mask
    masked_image = np.where(mask[..., None], [0, 1, 0], [0, 0, 0])  # Green color for mask
    ax.imshow(masked_image, alpha=0.3)  # Overlay with transparency

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Save the figure
plt.savefig("here.jpg", bbox_inches='tight', dpi=300)
plt.close()