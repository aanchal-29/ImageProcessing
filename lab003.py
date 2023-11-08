# Write a program to implement histogram equalization without using inbuilt function.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)

# Get image dimensions
height, width = image.shape

# Calculate histogram
histogram = np.zeros(256)
for i in range(height):
    for j in range(width):
        intensity = image[i, j]
        histogram[intensity] += 1

# Calculate cumulative distribution function (CDF)
cdf = np.zeros(256)
cdf[0] = histogram[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + histogram[i]

# Normalize CDF to the range of intensity values
cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

# Create equalized image
equalized_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        intensity = image[i, j]
        equalized_image[i, j] = cdf_normalized[intensity]

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plt.tight_layout()
plt.show()
