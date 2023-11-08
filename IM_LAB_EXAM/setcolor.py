import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to RGB color space
image = plt.imread('img/SET-C.png')
image_rgb = image.astype(np.uint8)

# Create histogram arrays for each color channel
histogram_r = np.zeros(256, dtype=int)
histogram_g = np.zeros(256, dtype=int)
histogram_b = np.zeros(256, dtype=int)

# Iterate over each pixel and update the histogram arrays
for row in image_rgb:
    for pixel in row:
        r, g, b = pixel
        histogram_r[r] += 1
        histogram_g[g] += 1
        histogram_b[b] += 1

# Plot the histograms
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.bar(range(256), histogram_r, color='red')
plt.title('Red Channel')

plt.subplot(1, 3, 2)
plt.bar(range(256), histogram_g, color='green')
plt.title('Green Channel')

plt.subplot(1, 3, 3)
plt.bar(range(256), histogram_b, color='blue')
plt.title('Blue Channel')

plt.tight_layout()
plt.show()
