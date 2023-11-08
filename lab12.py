# 12. Write a program to color an image and perform read and write operation.
import cv2

img = cv2.imread('D:/vs_workspace/imgPro_6th_sem/img/salt.png')

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to color
img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
# Display the input and colorized images
cv2.imshow('Input Image', img_gray)
cv2.waitKey(0)
cv2.imshow('Colorized Image', img_color)
cv2.waitKey(0)


# Write the colorized image to a file
# cv2.imwrite('D:/vs_workspace/imgPro_6th_sem/img/run_image.jpg', img_color)

# Read the colorized image from the file
img_out = cv2.imread('run_image.jpg')

# Display the read image
cv2.imshow('ReadImage', img_out)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
