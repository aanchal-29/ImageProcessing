# Write a program for resizing the image using inbuilt commands and without
# using inbuilt commands.
import cv2
import numpy as np

colorImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_COLOR)
cv2.imshow('colorImg ', colorImg )
new_size = (640, 480)

# Compute the scaling factor
x_scale = new_size[0] /colorImg.shape[1]
y_scale = new_size[1] /colorImg.shape[0]

# Create an empty image with the new size
resized_img = np.zeros((new_size[1], new_size[0], 3), dtype=np.uint8)

# Loop over the pixels of the new image and map them to the old image
for y in range(new_size[1]):
    for x in range(new_size[0]):
        old_x = int(x / x_scale)
        old_y = int(y / y_scale)
        resized_img[y,x,:] = colorImg[old_y,old_x,:]

# Display the resized image
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
