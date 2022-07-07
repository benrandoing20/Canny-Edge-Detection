# This module is aimed at performing Canny Edge Detection Analysis on a movie file
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
import pandas as pd
from datetime import datetime, timedelta


def find_ind(data_list, value):
    data_list = list(data_list)
    diff_list = []
    for i in data_list:
        diff_list.append(abs(i-value))
    min_diff = min(diff_list)
    index = diff_list.index(min_diff)
    return index, data_list[index]

def convert_pressure(value):
    return value / 51.7149 # from mmHg to psi

def can_main_window(vid_filename, csv_filename, lowP, highP):
    ### Move to Directory with Pressure and Image Data Inside
    # os.chdir("S:/Data/NPD - New Product Development/Compliance Lighting/Lt154 compliance/Lt154-1/Cycle 11")
    # os.chdir("C:/Users/brandoing/Documents/Canny-Edge-Detection/Lighting_Adjust")

    ### Read in Image Data and get File Start Time
    # vid_filename = 'Basler acA2000-165um (22709932)_20220621_140058344.avi'
    start_date = vid_filename.split('_')[-2]
    start_time = vid_filename.split('_')[-1].split('.')[0]
    start = start_date+start_time
    # print(start_time)
    creation_time = datetime.strptime(start, '%Y%m%d%H%M%S%f')
    # c_time = os.path.getmtime(filename)
    # # convert creation timestamp into DateTime object
    # creation_time = datetime.fromtimestamp(c_time)
    # print('Created on:', creation_time)
    # vid_start_time = creation_time - timedelta(seconds=8.01)
    # # print(vid_start_time)
    video = cv.VideoCapture(vid_filename)

    ### Read in Pressure Data
    # pData = pd.read_csv("Lt154 1 cycle 11 21JUN2022.csv", skiprows=2, skipfooter=3)
    pData = pd.read_csv(csv_filename, skiprows=2, skipfooter=3)
    pData = pData[['Date Time', 'Ch2 (psi)', 'Ch3 (psi)']]
    # print(pData.head())

    pArrayF = pData["Ch3 (psi)"].to_numpy()
    pArrayB = pData["Ch2 (psi)"].to_numpy()
    times = pData["Date Time"].to_numpy()
    # print(times[0])

    ## Find Exact Pressure and Indexes for 50 - 90 mmHg

    ind50, p50 = find_ind(pArrayF, convert_pressure((50)))
    time50 = times[ind50]
    date_time50 = datetime.strptime(time50, '%m/%d/%Y %H:%M:%S.%f')

    ind90, p90 = find_ind(pArrayF, convert_pressure((90)))
    time90 = times[ind90]
    date_time90 = datetime.strptime(time90, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Differences between Video Start and 50/90 Points
    dt50 = abs(date_time50 - creation_time)
    dt90 = abs(date_time90 - creation_time)

    ## Find Exact Pressure and Indexes for 80 - 120 mmHg

    ind80, p80 = find_ind(pArrayF, convert_pressure((80)))
    time80 = times[ind80]
    date_time80 = datetime.strptime(time50, '%m/%d/%Y %H:%M:%S.%f')

    ind120, p120 = find_ind(pArrayF, convert_pressure((120)))
    time120 = times[ind120]
    date_time120 = datetime.strptime(time120, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Differences between Video Start and 80/120 Points
    dt80 = abs(date_time80 - creation_time)
    dt120 = abs(date_time120 - creation_time)

    ## Find Exact Pressure and Indexes for 110 - 150 mmHg

    ind110, p110 = find_ind(pArrayF, convert_pressure((110)))
    time110 = times[ind110]
    date_time110 = datetime.strptime(time110, '%m/%d/%Y %H:%M:%S.%f')

    ind150, p150 = find_ind(pArrayF, convert_pressure((150)))
    time150 = times[ind150]
    date_time150 = datetime.strptime(time150, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Differences between Video Start and 110/150 Points
    dt110 = abs(date_time110 - creation_time)
    dt150 = abs(date_time150 - creation_time)

    ## Find Exact Pressure and Indexes for Low -High mmHg

    indLow, pLow = find_ind(pArrayF, convert_pressure((lowP)))
    timeLow = times[indLow]
    date_timeLow = datetime.strptime(timeLow, '%m/%d/%Y %H:%M:%S.%f')

    indHigh, pHigh = find_ind(pArrayF, convert_pressure((highP)))
    timeHigh = times[indHigh]
    date_timeHigh = datetime.strptime(timeHigh, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Differences between Video Start and Low/High Entry Points
    dtLow = abs(date_timeLow - creation_time)
    dtHigh = abs(date_timeHigh - creation_time)


    ## Find the Start and End Frames in the Video
    fps = 100  # Hz
    frame50 = round(dt50.total_seconds() * fps)
    frame90 = round(dt90.total_seconds() * fps)

    frame80 = round(dt80.total_seconds() * fps)
    frame120 = round(dt120.total_seconds() * fps)

    frame110 = round(dt110.total_seconds() * fps)
    frame150 = round(dt150.total_seconds() * fps)

    frameLow = round(dtLow.total_seconds() * fps)
    frameHigh = round(dtHigh.total_seconds() * fps)

    ## Find Maximum Pressure and Corresponding Timestamp
    max_p = np.max(pArrayF)
    max_index = np.where(pArrayF == max_p)[0][0]
    max_time = times[max_index]
    print(max_time)
    max_date_time = datetime.strptime(max_time, '%m/%d/%Y %H:%M:%S.%f')


    ## Find Minimum Pressure and Corresponding Timestamp
    first_deriv = np.gradient(pArrayF)
    min_index = np.where(first_deriv != 0)[0][0]
    min_p = pArrayF[min_index]
    min_time = times[min_index]
    print(min_time)
    min_date_time = datetime.strptime(min_time, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Difference between Min and Max Pressures
    dt = max_date_time - min_date_time

    ## Find the Pressure Difference
    dP = max_p - min_p

    ## Find Time Differences between Video Start and Min/Max P
    dt_min = abs(min_date_time - creation_time)
    dt_max = abs(max_date_time - creation_time)

    print(dt_min)
    print(dt_max)

    ## Find the Start and End Frames in the Video
    fps = 100  # Hz
    frame_start = round(dt_min.total_seconds()*fps)
    frame_end = round(dt_max.total_seconds()*fps)
    print(frame_start)
    print(frame_end)

    ## Prepare Changing Diameter
    diameter_v_time = []



    ## Find HAV Pixel Diameter


    frame = 1
    while(video.isOpened()):

        success, image = video.read()
        edges = cv.Canny(image, 30, 40)


        if success:

            left_edges = []
            right_edges = []
            count = 0
            for row in edges:
                left_edges.append(np.nonzero(row)[0][0])
                right_edges.append(np.nonzero(row)[0][-1])
                # for pixel in row:
                #     if pixel != 0:
                #         left_edges.append(list(row).index(pixel))
                #         break

                # list(row).reverse()
                # for pixel in row:
                #     if pixel != 0:
                #         indx = list(row).index(pixel)
                #         right_edges.append((len(row)-indx-1))
                #         break

            left_edge = np.mean(left_edges)
            right_edge = np.mean(right_edges)
            diff = np.array(right_edges)-np.array(left_edges)
            pixel_diameter = right_edge-left_edge
            diameter_v_time.append(pixel_diameter)
            print(pixel_diameter)
            frame +=1
        else:
            break
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
    video.release()
    cv.destroyAllWindows()

    ### Find pixel_diameters at Frames of Interest

    start_diameter = diameter_v_time[frame_start]
    try:
        end_diameter = diameter_v_time[frame_end]
    except IndexError:
        end_diameter = diameter_v_time[-1]

    ### Find pixel_diameters at Frames 50/90 80/120 110/150

    diameter50 = diameter_v_time[frame50]
    diameter90 = diameter_v_time[frame90]

    diameter80 = diameter_v_time[frame80]
    diameter120 = diameter_v_time[frame120]

    diameter110 = diameter_v_time[frame110]
    diameter150 = diameter_v_time[frame150]

    diameterLow = diameter_v_time[frameLow]
    diameterHigh = diameter_v_time[frameHigh]

    ### Calculate Compliance Min/Max

    vessel_thickness = 0
    start_radius = start_diameter/2 - vessel_thickness
    end_radius = end_diameter/2 - vessel_thickness
    compliance = ((end_radius - start_radius)/start_radius) / (dP*51.7149) * 10000
    print(compliance)

    ### Calculate Compliance Low/High GUI Entry

    vessel_thickness = 0
    start_radius = diameterLow / 2 - vessel_thickness
    end_radius = diameterHigh / 2 - vessel_thickness
    complianceLH = ((end_radius - start_radius) / start_radius) / ((pHigh - pLow) * 51.7149) * 10000
    print(complianceLH)

    ### Calculate Compliance 50/90

    vessel_thickness = 0
    start_radius = diameter50 / 2 - vessel_thickness
    end_radius = diameter90 / 2 - vessel_thickness
    compliance5090 = ((end_radius - start_radius) / start_radius) / ((p90-p50) * 51.7149) * 10000
    print(compliance5090)

    ### Calculate Compliance 80/120

    vessel_thickness = 0
    start_radius = diameter80 / 2 - vessel_thickness
    end_radius = diameter120 / 2 - vessel_thickness
    compliance80120 = ((end_radius - start_radius) / start_radius) / ((p120 - p80) * 51.7149) * 10000
    print(compliance80120)
    # return compliance80120

    ### Calculate Compliance 110/150

    vessel_thickness = 0
    start_radius = diameter110 / 2 - vessel_thickness
    end_radius = diameter150 / 2 - vessel_thickness
    compliance110150 = ((end_radius - start_radius) / start_radius) / ((p150 - p110) * 51.7149) * 10000
    print(compliance110150)
    return compliance5090, compliance80120, compliance110150, complianceLH


    ########################## Plotting ########################################

    ### Plots the Numerical Edge Indexes Left and Right
    # plt.figure()
    # plt.plot(left_edges)
    # plt.plot(right_edges)
    # plt.plot(diff)
    # plt.legend(["left", "right", "diff"])
    # plt.show()

    ### Plots the Pixel Diameter over Time
    # plt.figure()
    # plt.plot(diameter_v_time)
    # plt.savefig("HAV_diameter_LT154_1_Cycle11_Updated.jpg")
    # plt.show()

    ### Plots the images
    # plt.figure(1, figsize = [12,8])
    # plt.subplot(1,2,1)
    # plt.imshow(image)
    # # plt.title("Ring_Centered_7_4")

    # plt.subplot(1,2,2)
    # plt.imshow(edges+50, cmap = "gray")
    # # plt.title("Ring_Centered_7_4_Edges")
    # # plt.savefig("Ring_Centered_7_4_Data.jpg")
    # plt.show()


    ###################################### Exporting ##################################

    ### Save Data to CSV File

if __name__ == '__main__':
    os.chdir("S:/Data/NPD - New Product Development/Compliance Lighting/Lt154 compliance/Lt154-2/Cycle 11")
    video_file = 'Basler acA2000-165um (22709932)_20220621_142115740.avi'
    pressure_file = "Lt154-2 cycle 11 21JUN2022.csv"
    compliance = can_main_window(video_file, pressure_file, 50, 90)



