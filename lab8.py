# Write a program to show the non-linear filtering technique using edge detection.
import cv2

# Load the input image
img = cv2.imread('D:/vs_workspace/imgPro_6th_sem/img/lake.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(blur, 50, 150)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
