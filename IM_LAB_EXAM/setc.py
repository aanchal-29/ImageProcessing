
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Gray Image
color_image=cv2.imread("img/SET-C.png")
gray_image = cv2.imread("img/SET-C.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("color_image",color_image)
cv2.imshow("gray",gray_image)
cv2.waitKey(0)
height, width = gray_image.shape[:2]

# giving 256 as its range is from [0,255], creating array instialized with 0
histogram = np.zeros(256)
# counting the frequency of pixels
for i in range(height):
    for j in range(width):
        intensity = gray_image[i, j]
        histogram[intensity] += 1

# Calculate cumulative distribution function -> summing the freq
cdf = np.zeros(256)
cdf[0] = histogram[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + histogram[i] 

cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  


# equalized image
equalized_image = np.zeros_like(gray_image)
for i in range(height):
    for j in range(width):
        intensity = gray_image[i,j]
        equalized_image[i,j] = cdf_normalized[intensity]



print("Histogram values:", equalized_image)
gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])


plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title(' Grayscale Histogram')
plt.show()

color_hist_r = cv2.calcHist([color_image], [0], None, [256], [0, 256])
color_hist_g = cv2.calcHist([color_image], [1], None, [256], [0, 256])
color_hist_b = cv2.calcHist([color_image], [2], None, [256], [0, 256])

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

plt.tight_layout()
plt.show()

