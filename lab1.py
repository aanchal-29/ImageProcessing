# pip install opencv-python cv2, numpy and matplotlib libraries
# pip install opencv-python
# pip install numpy
# pip install matplotlib

# Matplotlib library uses RGB color format to read a colored image. 
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("D:/vs_workspace/imgPro_6th_sem/img/lake.png")
#Displaying image using plt.imshow() method
plt.imshow(img)

#hold the window
plt.waitforbuttonpress()
plt.close('all')
