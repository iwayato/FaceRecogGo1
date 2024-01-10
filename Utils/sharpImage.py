import matplotlib.pyplot as plt 
import numpy as np
import cv2 as cv

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

img = cv.imread("./Frames/4m.png")
sharpened = cv.filter2D(img, -1, kernel)

plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(img) 

plt.subplot(1, 2, 2) 
plt.title("Sharpening") 
plt.imshow(sharpened) 
plt.show()