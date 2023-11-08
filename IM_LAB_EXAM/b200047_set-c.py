
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image):
    
    histogram = np.zeros(256, dtype=int)

   
    for row in image:
        for pixel in row:
            intensity = pixel
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


image = plt.imread('img/SET-C.png')
grayscale_image =  cv2.imread("img/SET-C.png", cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
histogram = compute_histogram(grayscale_image)
#plot_histogram(histogram)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blue_hist, green_hist, red_hist = compute_color_histogram(rgb_image)
plot_color_histogram(blue_hist, green_hist, red_hist)
