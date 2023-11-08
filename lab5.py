# 5a). Write a program for resizing the image using inbuilt commands and without
# using inbuilt commands.
import cv2

# Load the image

colorImg = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_COLOR)
cv2.imshow('colorImg ', colorImg )
# Set the new size
# h, w = colorImg.shape[:2]
# new_h, new_w = int(h / 2), int(w / 2)

new_size = (640, 480)

# Resize the image
resized_img = cv2.resize(colorImg, new_size)

# Display the resized image
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
