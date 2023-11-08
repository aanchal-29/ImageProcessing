# Q11. Write a program to discretize an image using Fourier transform.
import cv2
import numpy as np

# Read the image
img = cv2.imread('D:/vs_workspace/imgPro_6th_sem/img/lake.png', 0)

# Perform FFT on the image
fft_img = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fft_img = np.fft.fftshift(fft_img)

# Define the frequency cut-off value
cutoff = 50

# Set the high-frequency coefficients to zero
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
fft_img[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

# Shift the zero-frequency component back to the top-left corner of the spectrum
fft_img = np.fft.ifftshift(fft_img)

# Perform IFFT on the image
ifft_img = np.fft.ifft2(fft_img)

# Normalize the pixel values of the output image
out_img = np.abs(ifft_img) / np.max(np.abs(ifft_img)) * 255

# Convert the output image to uint8 data type
out_img = np.uint8(out_img)

# Display the input and output images
cv2.imshow('Input Image', img)
cv2.imshow('Output Image', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
