# Write a program to remove Salt and Pepper noise using median filter.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('D:/vs_workspace/imgPro_6th_sem/img/lake.png', cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise
noise_img = img.copy()
noise_mask = np.random.choice((0, 1, 2), size=img.shape, p=[0.9, 0.05, 0.05])
noise_img[noise_mask == 1] = 255
noise_img[noise_mask == 2] = 0

# Apply median filter to remove noise
filtered_img = cv2.medianBlur(noise_img, 3)

# Display the original and filtered images side by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(noise_img, cmap='gray')
ax[0].set_title('Noisy Image')
ax[0].axis('off')
ax[1].imshow(filtered_img, cmap='gray')
ax[1].set_title('Filtered Image')
ax[1].axis('off')
plt.show()
