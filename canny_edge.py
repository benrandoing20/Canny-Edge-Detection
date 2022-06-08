# This module is aimed at performing Canny Edge Detection Analysis on a movie file

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os

os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/PTFE 04APR2022/new ptfe 70S06/Sample 1 - Run 1")

video = cv.VideoCapture("Basler acA2000-165um (22709932)_20220404_114506150.avi")

success, image = video.read()

print(image)

#cv.Canny(image)



plt.figure()
plt.imshow(image)
plt.title("Frame 1")
plt.show()


