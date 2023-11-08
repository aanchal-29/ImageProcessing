# 10. Write a program to eliminate the high frequency component of an image.
import cv2
import numpy as np

# Load the image
img = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Subtract the blurred image from the original to obtain the high-frequency component
high_freq = gray - blur

# Set the high-frequency component to zero
gray_filtered = gray - high_freq

# Show the original and filtered images
cv2.imshow('Original', gray)
cv2.imshow('Filtered', gray_filtered)
cv2.waitKey(0)