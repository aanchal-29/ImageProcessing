# Write a program to perform edge detection using different operators.
import cv2
import numpy as np

# Load the image
grayImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)
 
img = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)

# Define the kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Apply the kernels to the image
edges_x = cv2.filter2D(img, -1, sobel_x)
edges_y = cv2.filter2D(img, -1, sobel_y)
edges_lap = cv2.filter2D(img, -1, laplacian)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Sobel X', edges_x)
cv2.imshow('Sobel Y', edges_y)
cv2.imshow('Laplacian', edges_lap)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
