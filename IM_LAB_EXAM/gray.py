
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image):
    # Initialize an array to store the histogram values
    histogram = np.zeros(256, dtype=int)

    # Iterate over each pixel in the image
    for row in image:
        for pixel in row:
            # Get the pixel intensity value
            intensity = pixel

            # Increment the corresponding histogram bin
            histogram[intensity] += 1

    return histogram

def plot_histogram(histogram):
    # Create an array of bin values from 0 to 255
    bins = np.arange(256)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.title('Grayscale Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.bar(bins, histogram, color='gray', alpha=0.7)
    plt.show()

# Example usage
image = plt.imread('img/SET-C.png')
grayscale_image =  cv2.imread("img/SET-C.png", cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
histogram = compute_histogram(grayscale_image)
plot_histogram(histogram)
