import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_color_histogram(image):
    # Split the image into color channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # Compute the histogram for each color channel
    blue_hist = np.zeros(256, dtype=int)
    green_hist = np.zeros(256, dtype=int)
    red_hist = np.zeros(256, dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blue_hist[blue_channel[i, j]] += 1
            green_hist[green_channel[i, j]] += 1
            red_hist[red_channel[i, j]] += 1

    return blue_hist, green_hist, red_hist

def plot_color_histogram(blue_hist, green_hist, red_hist):
    # Plot the histograms
    plt.figure(figsize=(10, 6))
    plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(blue_hist, color='blue', label='Blue')
    plt.plot(green_hist, color='green', label='Green')
    plt.plot(red_hist, color='red', label='Red')
    plt.legend()
    plt.show()

# Example usage
image = cv2.imread('img/SET-C.png')

# Convert the image to RGB if it's in BGR format
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Compute the color histograms
blue_hist, green_hist, red_hist = compute_color_histogram(rgb_image)

# Plot the histograms
plot_color_histogram(blue_hist, green_hist, red_hist)
