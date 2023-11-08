# Python code to read image
# 1. Write a program to display a grayscale image using read and write operation.


import cv2
import numpy as np
from matplotlib import pyplot as plt
# To read image from disk, we use
# cv2.imread function, in below method,
colorImg = cv2.imread("img/lake.png", cv2.IMREAD_COLOR)

h, w = colorImg.shape[:2]
# Displaying the height and width
print("Height = {},  Width = {}".format(h, w))


plt.subplot(2, 2, 1)

cv2.imshow("ColorImage", colorImg)
cv2.waitKey(0)

grayImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)

plt.subplot(2, 2, 2)
# Displaying the image
cv2.imshow('GrayScaleImage', grayImg)
cv2.waitKey(0)


# plt.show()
cv2.destroyAllWindows()

# cv2.imwrite('output_image.jpg', image)

