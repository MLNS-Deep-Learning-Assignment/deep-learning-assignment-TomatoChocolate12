import cv2
import numpy as np
import os

# Set paths for input and output
input_dir = '../only_data'  # Replace with the path to your .npy files
output_images_dir = '../../../../scratch/yashas.b/segmented'  # Where to save cropped images
output_labels_dir = '../../../../scratch/yashas.b/segmented_labels'  # Where to save annotations

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Function to generate annotations for YOLO format
def generate_annotations(image, image_filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotations = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Skip small contours (non-digit)
            continue
        x, y, w, h = cv2.boundingRect(contour)
        height, width = image.shape[:2]
        
        # Normalize the coordinates for YOLO format
        x_center = (x + x + w) / 2 / width
        y_center = (y + y + h) / 2 / height
        width_normalized = w / width
        height_normalized = h / height
        
        annotations.append([0, x_center, y_center, width_normalized, height_normalized])  # Class 0 for all digits
    
    return annotations

# Process each .npy file
for npy_filename in os.listdir(input_dir):
    if npy_filename.endswith('.npy'):
        npy_path = os.path.join(input_dir, npy_filename)
        
        # Load the .npy data (assumed to be a 3D array)
        data = np.load(npy_path)  # Shape (10000, 40, 168)

        for idx, image in enumerate(data):
            # Convert the image from (40, 168) to a BGR image (for visualization and contour finding)
            image_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Generate bounding box annotations
            annotations = generate_annotations(image_bgr, npy_filename)

            # Save the image and annotations
            image_filename = f"{npy_filename.replace('.npy', '')}_{idx}.jpg"
            label_filename = f"{npy_filename.replace('.npy', '')}_{idx}.txt"

            cv2.imwrite(os.path.join(output_images_dir, image_filename), image_bgr)
            
            with open(os.path.join(output_labels_dir, label_filename), 'w') as label_file:
                for annotation in annotations:
                    label_file.write(' '.join(map(str, annotation)) + '\n')
