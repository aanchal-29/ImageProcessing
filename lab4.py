# 4. Write a program to extract the bit planes of a grayscale image.
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the grayscale image
# img = cv2.imread('grayscale_image.png', cv2.IMREAD_GRAYSCALE)
grayImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('GrayScaleImage', grayImg)
# Create an array to store the bit planes
bit_planes = np.zeros((8, grayImg.shape[0], grayImg.shape[1]), dtype=np.uint8)
# each pixel value is an unsigned 8-bit integer (i.e., a number in the range [0, 255]). 

# Extract the bit planes
for i in range(8):
    bit_planes[i] = np.bitwise_and(grayImg, 2**i)
    bit_planes[i] = bit_planes[i] * 255 / 2**i

# Display the bit planes
for i in range(8):
    cv2.imshow('Bit plane '+str(i), bit_planes[i])

# cc = (2 * (2 * (2 * (2 * (2 * (2 * (2 * bit_planes[7] + bit_planes[6]) +bit_planes[5]) +bit_planes[4]) + bit_planes[3]) + bit_planes[2]) + bit_planes[1]) +bit_planes[0]);

# cv2.imshow('Bit plane '+(cc));

recombined_img = np.zeros_like(grayImg)
for i in range(8):
    recombined_img += bit_planes[i] * 2**(7-i)

# Display the recombined image
plt.imshow(recombined_img, cmap='gray')
cv2.imshow('recombined_img',recombined_img)
# plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
