# 13. Write a program to segment an image using polynomial curve fitting.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)

# Define the degree of the polynomial
degree = 3

# Generate x and y coordinates of the image pixels
x = np.arange(0, img.shape[1])
y = np.arange(0, img.shape[0])
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Generate the intensity values of the image pixels
zz = img.flatten()

# Fit the polynomial curve to the intensity values
coefficients = np.polyfit(xx, yy, degree, w=zz)
p = np.poly1d(coefficients)

# Generate the segmented image
img_segmented = np.zeros_like(img)
for i in range(img.shape[1]):
    y_fit = p(i)
    img_segmented[int(y_fit), i] = img[int(y_fit), i]

# Display the input and segmented images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_segmented, cmap='gray')
plt.title('Segmented Image')
plt.axis('off')

plt.show()