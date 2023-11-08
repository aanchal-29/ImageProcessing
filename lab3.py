# Write a program to implement histogram equalization without using inbuilt function.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)

# Compute the histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Compute the cumulative distribution function
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Perform histogram equalization
equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized)

# Reshape the equalized image to its original size
equalized_img = equalized_img.reshape(img.shape)

# Display the original and equalized images side by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(equalized_img, cmap='gray')
ax[1].set_title('Equalized Image')
ax[1].axis('off')
plt.show()
