# This module is aimed at performing Canny Edge Detection Analysis on a movie file

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os

os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/PTFE 04APR2022/new ptfe 70S06/Sample 1 - Run 1")

video = cv.VideoCapture("Basler acA2000-165um (22709932)_20220404_114506150.avi")

frame_num = int(video.get(cv.CAP_PROP_FRAME_COUNT))
# print(frame_num)
diameter_v_time = []

# for i in range(1):
success, image = video.read()
edges = cv.Canny(image, 30, 60)
# print(edges)

left_edges = []
right_edges = []

for row in edges:
    for pixel in row:
        if pixel != 0:
            left_edges.append(list(row).index(pixel))
            break

    list(row).reverse()
    for pixel in row:
        if pixel != 0:
            indx = list(row).index(pixel)
            right_edges.append((len(row)-indx-1))
            break

left_edge = np.mean(left_edges)
right_edge = np.mean(right_edges)
pixel_diameter = right_edge-left_edge
diameter_v_time.append(pixel_diameter)


# plt.figure()
# plt.plot(diameter_v_time)
# plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Frame 1")


plt.subplot(1,2,2)
plt.imshow(edges+50, cmap = "gray")
plt.title("Edges 1")
plt.show()



