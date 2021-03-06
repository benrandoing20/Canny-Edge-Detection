# This module is aimed at performing Canny Edge Detection Analysis on a movie file to Yield Compliance
# Metrics for the compliance_gui.py code

# Created by Benjamin Randoing June-August 2022

### Import Libraries Implemented in Code
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
import pandas as pd
from datetime import datetime, timedelta


### This function takes a pressure input to find the index, timestamp of occurrence to identiy the video frame at the same time
def find_ind(data_list, value):
    data_list = list(data_list) # List of Pressure Values
    diff_list = []
    for i in data_list:
        diff_list.append(abs(i-value)) # Use Minimum difference between pressures and input value to find index
    min_diff = min(diff_list)
    index = diff_list.index(min_diff)
    return index, data_list[index]

### This function crops pressure data based on the video occurrence
def trim_pressure(pressure_data, time_data, video_start, video_length, fps_vid, fps_p=100):
    start_indx = 0
    start_time = datetime.strptime(time_data[0], '%m/%d/%Y %H:%M:%S.%f')
    diff = abs(start_time - video_start)
    for i in range(len(time_data)): # Investigates pressure timestamps with dT to start of video to find when to start pressure data
        formatted_times = datetime.strptime(time_data[i], '%m/%d/%Y %H:%M:%S.%f')
        local_diff = abs(formatted_times - video_start)
        if local_diff < diff:
            diff = local_diff
            start_indx = i

    pressure_data_trim = pressure_data[start_indx:start_indx+int(video_length/fps_vid*fps_p)]
    time_data_trim = time_data[start_indx:start_indx + int(video_length/fps_vid*fps_p)]
    return pressure_data_trim, time_data_trim

### This is the core function that finds compliance values provided frames, pressures, and an array of Canny Filtered Images
def get_mean_compliance(frame_low, frame_high, p_low, p_high, edges):
    left_edges_low = []
    right_edges_low = []
    # The first two loops find the edges for the frames corresponding to low and high pressures
    for row in edges[frame_low]:
        try:
            left_edges_low.append(np.nonzero(row)[0][0])
            right_edges_low.append(np.nonzero(row)[0][-1])
        except IndexError:
            left_edges_low.append(np.mean(left_edges_low))
            right_edges_low.append(np.mean(left_edges_low))
    left_edges_high = []
    right_edges_high = []
    for row in edges[frame_high]:
        try:
            left_edges_high.append(np.nonzero(row)[0][0])
            right_edges_high.append(np.nonzero(row)[0][-1])
        except IndexError:
            left_edges_high.append(np.mean(left_edges_high))
            right_edges_high.append(np.mean(right_edges_high))


        pixel_diameters_low = []
        pixel_diameters_high = []
        compliances = []

    low_med_diameter = np.median(right_edges_low) - np.median(left_edges_low)
    high_med_diameter = np.median(right_edges_high) - np.median(left_edges_high)

    # The following loop finds the pixel diameter of each frame and the calculates compliance
    for i in range(len(left_edges_low)):
        pixel_diameters_low.append(right_edges_low[i] - left_edges_low[i])
        pixel_diameters_high.append(right_edges_high[i] - left_edges_high[i])

        comp = (((pixel_diameters_high[i]/2) - (pixel_diameters_low[i]/2)) / pixel_diameters_low[i]/2) / ((p_high - p_low) * 51.7149) * 10000
        comp_v2 = (((high_med_diameter/2) - (low_med_diameter/2)) / low_med_diameter/2) / ((p_high - p_low) * 51.7149) * 10000
        compliances.append(comp)

    std_comp = np.std(compliances)
    mean_comp = np.mean(compliances)

    fixed_compliances = []
    for comp in compliances:
        if comp > mean_comp+std_comp or comp < mean_comp-std_comp:
            continue
        elif comp == None:
            continue
        fixed_compliances.append(comp)

    y_vals = list(range(1088))
    # The mean value of compliance is found using the median to avoid influence from significant outliers
    mean_compliance = np.median(fixed_compliances)

    # The following plots are the median line through all calculated compliances and the visualization of the edges used to
    # debug the canny Edge Detection Threshold Values
    plt.figure()
    plt.plot(fixed_compliances)
    plt.axhline(y=mean_compliance)
    plt.show()

    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(edges[frame_low], cmap = 'gray')
    plt.plot(left_edges_low, y_vals, color='r', linestyle='-')
    plt.plot(right_edges_low, y_vals, color='r', linestyle='-')

    plt.subplot(1,2,2)
    plt.imshow(edges[frame_high], cmap = 'gray')
    plt.plot(left_edges_high, y_vals, color='r', linestyle='-')
    plt.plot(right_edges_high, y_vals, color='r', linestyle='-')
    plt.show()

    return mean_compliance

# This function converts pressure from mmHg to psi
def convert_pressure(value):
    return value / 51.7149 # from mmHg to psi

# This is the main function called by the gui
def can_main_window(vid_filename, csv_filename, lowP, highP, filepath):

    ### Read in Image Data and get File Start Time
    start_date = vid_filename.split('_')[-2]
    start_time = vid_filename.split('_')[-1].split('.')[0]
    start = start_date+start_time
    # creation_time = datetime.strptime(start, '%Y%m%d%H%M%S%f')
    video = cv.VideoCapture(vid_filename)

    ### Read in Pressure Data
    pData = pd.read_csv(csv_filename, skiprows=2, skipfooter=3)
    pData = pData[['Date Time', 'Ch4 (psi)', 'Ch3 (psi)']]
    # print(pData.head())

    pArrayF_raw = pData["Ch3 (psi)"].to_numpy()
    pArrayB_raw = pData["Ch4 (psi)"].to_numpy()
    times_raw = pData["Date Time"].to_numpy()

    ### ALternate Code to find Start Time if Video Timestamp is Corrupted
    threshold = abs(pArrayF_raw[0])*1.5
    creation_ind, creation_p = find_ind(pArrayF_raw, threshold)
    creation_time_bare = times_raw[creation_ind]
    creation_time = datetime.strptime(creation_time_bare, '%m/%d/%Y %H:%M:%S.%f')

    fps_vid = video.get(cv.CAP_PROP_FPS)
    frames = (video.get(cv.CAP_PROP_FRAME_COUNT)-1)
    pArrayF, times = trim_pressure(pArrayF_raw, times_raw, creation_time, frames, fps_vid)

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
    date_time80 = datetime.strptime(time80, '%m/%d/%Y %H:%M:%S.%f')

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
    fps = fps_vid  # Hz
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
    # print(max_time)
    max_date_time = datetime.strptime(max_time, '%m/%d/%Y %H:%M:%S.%f')


    ## Find Minimum Pressure and Corresponding Timestamp
    first_deriv = np.gradient(pArrayF)
    min_index = np.where(first_deriv != 0)[0][0]
    min_p = pArrayF[min_index]
    min_time = times[min_index]
    # print(min_time)
    min_date_time = datetime.strptime(min_time, '%m/%d/%Y %H:%M:%S.%f')

    ## Find Time Difference between Min and Max Pressures
    dt = max_date_time - min_date_time

    ## Find the Pressure Difference
    dP = max_p - min_p

    ## Find Time Differences between Video Start and Min/Max P
    dt_min = abs(min_date_time - creation_time)
    dt_max = abs(max_date_time - creation_time)

    # print(dt_min)
    # print(dt_max)

    ## Find the Start and End Frames in the Video
    fps = fps_vid  # Hz
    frame_start = round(dt_min.total_seconds()*fps)
    frame_end = round(dt_max.total_seconds()*fps)
    # print(frame_start)
    # print(frame_end)


    ## Find HAV Pixel Diameter
    frame = 0
    all_frames = []
    all_edges = []

    ### Loop Through ALl Frames to Obtain a list of Canny Edge Filtered Images
    while(video.isOpened()):
        success, image = video.read()
        if not success:
            break
        print(image[0])
        edges = cv.Canny(image, 20,40)
        all_frames.append(image)
        all_edges.append(edges)
        frame += 1
        print(frame)

    video.release()
    cv.destroyAllWindows()

    outputs = []
    outputs.append(get_mean_compliance(frameLow, frameHigh, pLow, pHigh, all_edges))
    outputs.append(get_mean_compliance(frame50, frame90, p50, p90, all_edges))
    outputs.append(get_mean_compliance(frame80, frame120, p80, p120, all_edges))
    outputs.append(get_mean_compliance(frame110, frame150, p110, p150, all_edges))


    ###################################### Exporting ##################################

    ### Save Data to CSV File

    date_time_now = datetime.now()
    string_out_time = date_time_now.strftime('%d%b%Y_%H%M%S')


    df1 = pd.DataFrame({'TimeStamp': times,
                        'Pressure (psi)': pArrayF,
                        'Compliance 50/90': outputs[1],
                        'Compliance 80/120': outputs[2],
                        'Compliance 110/150': outputs[3]})

    os.chdir(filepath)
    df1.to_csv('Data_Output_'+string_out_time+'.csv', index=False)

    return outputs



# This runs the code if the gui is not run. 
if __name__ == '__main__':
    os.chdir("C:/Users/brandoing/Desktop/Lt154-8/cycle 11")
    video_file = 'Basler acA2000-165um (22709932)_20220711_130243837.avi'
    pressure_file = "Lt154-8 cycle 11 11JULY2022.csv"
    filepath = os.path.dirname(os.path.abspath(video_file))
    compliance = can_main_window(video_file, pressure_file, 50, 90, filepath)



