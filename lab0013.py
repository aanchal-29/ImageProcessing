# 13. Write a program to segment an image using polynomial curve fitting.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('img/lake.png', cv2.IMREAD_GRAYSCALE)

# Extract the image data as a 1D array
image_data = image.ravel()

# Define the degree of the polynomial curve
degree = 3

# Perform polynomial curve fitting
coefficients = np.polyfit(np.arange(len(image_data)), image_data, degree)

# Generate the polynomial curve
curve = np.polyval(coefficients, np.arange(len(image_data)))

# Threshold the image based on the curve
threshold = 0.5
segmented_image = np.where(image_data > curve * threshold, 255, 0)

# Reshape the segmented image back to its original shape
segmented_image = segmented_image.reshape(image.shape)

# Display the segmented image
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')
plt.show()
