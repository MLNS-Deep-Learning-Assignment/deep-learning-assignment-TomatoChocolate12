import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "../../../../scratch/yashas.b/images/data0_324.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Invert the image (make digits white and background black)
image = cv2.bitwise_not(image)

# Apply adaptive thresholding for better digit separation
thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Dilate the image to connect fragmented parts of digits
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by x-coordinate to maintain digit order
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

# Extract individual digits
digit_images = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filter out small contours that are likely noise
    if w > 5 and h > 5:  # Adjust these thresholds based on your data
        digit = image[y:y+h, x:x+w]
        digit_images.append(digit)

# Display extracted digits
for i, digit in enumerate(digit_images):
    plt.subplot(1, len(digit_images), i+1)
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
plt.title(f"Digit {i}")
plt.tight_layout()
plt.savefig("h.png")

# Optionally save each digit
for i, digit in enumerate(digit_images):
    cv2.imwrite(f"extracted_digits/digit_{i}.png", digit)
