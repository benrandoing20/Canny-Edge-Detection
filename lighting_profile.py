# This module is intended to plot the HAV profile with various lighting

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os


os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/Lighting_Adjust")
files = os.listdir()

plt.figure(1, figsize=[12,8])
for i, file in enumerate(files):
    if ".jpg" in file:
        if "Rect" in file:
            filename = file
            image = cv.imread(filename)


            height, width = image.shape[:2]
            # print("The image height is {} and width is {}".format(height, width))


            first_quarter = int(height/4)
            second_quarter = int(height/2)
            third_quarter = int(height/4*3)


            first_quarter_row = image[first_quarter,:]
            second_quarter_row = image[second_quarter,:]
            third_quarter_row = image[third_quarter,:]


            legend_labels = ["px 272", "px 544", "px 816"]



            fig = plt.subplot(2,3,i-1)
            plt.plot(first_quarter_row)
            plt.plot(second_quarter_row)
            plt.plot(third_quarter_row)
            plt.title(filename.removesuffix(".jpg"))
            # plt.xlabel("pixel")
            # plt.ylabel("Signal Amplitude")
            # plt.legend(legend_labels)
            #
            # plt.subplot(1,2,2)
            # plt.imshow(image, cmap = 'gray', aspect = "auto")
            # plt.title(filename.removesuffix(".jpg")+"_Raw")

plt.tight_layout()
os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/Lighting_Adjust/Data Plots")
plt.savefig("All_Rect_Lighting_Profiles.jpg")
# plt.savefig(filename.removesuffix(".jpg") + "_Dataplot.jpg")
os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/Lighting_Adjust")
plt.show()

