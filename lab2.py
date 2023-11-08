# 2. Write a program to find the histogram value and display the histogram of a grayscale and color image.


import cv2
import numpy as np
from matplotlib import pyplot as plt


# fot grayscale img

# grayImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/Sir3/image1.tif", 0)
grayImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('GrayScaleImage', grayImg)
cv2.waitKey(0)
# Calculate histogram values
gray_hist = cv2.calcHist([grayImg], [0], None, [256], [0, 256])

# Plot histogram
plt.hist(grayImg.ravel(), 256, [0, 256])
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title(' Grayscale Histogram')
plt.show()
# Print histogram values
print("Histogram values:", gray_hist)


# for Color img

color_img = cv2.imread('D:/vs_workspace/imgPro_6th_sem/img/lake.png')
color_hist_r = cv2.calcHist([color_img], [0], None, [256], [0, 256])
color_hist_g = cv2.calcHist([color_img], [1], None, [256], [0, 256])
color_hist_b = cv2.calcHist([color_img], [2], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.plot(gray_hist, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')

plt.subplot(2, 2, 2)
plt.plot(color_hist_r, color='red')
plt.plot(color_hist_g, color='green')
plt.plot(color_hist_b, color='blue')
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.legend(['Red Channel', 'Green Channel', 'Blue Channel'])

plt.show()
cv2.destroyAllWindows()

