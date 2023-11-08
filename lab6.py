# Write a program to implement high pass first order filter.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image

img = cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png", cv2.IMREAD_GRAYSCALE)
height, width = img.shape
# Define the high-pass filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Convolve the image with the kernel
filtered_img = cv2.filter2D(img, -1, kernel)
print("The matrix of the original image:")
  
  
for i in range(0, height):
    for j in range(0, width):
        print(img[i][j], end=" ")
    print()

print("The matrix form after HPF masking the captured image is:")
print("\n")
for hpf in filtered_img:
    print(hpf)

# Display the original and filtered images side by side
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(filtered_img, cmap='gray')
ax[1].set_title('Filtered Image')
ax[1].axis('off')
plt.show()
