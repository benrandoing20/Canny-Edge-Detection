# Compliance Analysis
### Created by Benjamin Randoing
### email: benrandoing20@gmail.com
### phone: (214) 356-9043

## Purpose
Following conversations with surgeons, Humacyte recognizes the ideal of a more compliant vessel. Provided the 
 limitation of the inability to incorporate biofidelic quantities of elastin as part of the HAV, a reliable metric to quantify 
compliance is a valuable asset in comparing HAVs to existing ePTFE grafts. 

This code streamlines the analysis of video and pressure data to quantify vessel compliance in accordance with ISO 7198.

## Code Installation
1. Python Must be installed on the Computer. The code was designed using Python 3.10.5, which can be downloaded[ here](https://www.python.org/downloads/).
2. It is recommended to have git bash terminal installed on the computer
   1. This can be downloaded [here](https://git-scm.com/). 
   2. During installation, accept the preselected options with the following exceptions:
      1. On the Select Components screen, I deselect Git Bash Here and Git GUI Here under Windows Explorer integration to prevent cluttering my right-click context menu. But, you may leave if desired. 
      2. On the Choosing the default editor used by Git screen, I would select a different editor rather than Vim. I like Notepad++, but standard Notepad is a good choice, also. 
      3. On the Adjusting the name of the initial branch in new repositories screen, select Override the default branch name for new repositories and enter main as the new default name. This will now mirror the behavior of GitHub. 
      4. On the Configuring the line ending conversions screen, the default should be Checkout Windows-style, commit Unix-style line endings. Make sure that it is.
3. Ensure to have the **compliance_gui.py**, **canny_edges.py**, and **requirements.txt** files in the same directory
4. In git bash, use `cd` to navigate to the directory with the aforementioned python scripts. 
5. A virtual environment must be created by typing `python -m venv venv` in a git bash terminal.
6. Navigate into the new virtual environment by typing `source venv/Scripts/activate`
7. Type `pip install -r requirements.txt` to import the necessary libraries.
   1. numpy is imported for basic math and the use of numpy arrays
   2. opencv-python is imported to process the video avi data
   3. matplotlib is imported to plot the automatically detected edges when the gui is run
   4. datetime is imported to use the real time to align time and pressure data
   5. scikit-learn is imported to train KMeans models and to perform PCA
   6. pandas is installed to use the Dataframe ata structure with csv files

## How to make GUI executable
1. Ensure you are in the virtual environment and the correct folder in git bash. 
2. Type `pip install pyinstaller`.
3. Type `pyinstaller --onefile -w compliance_gui.py`

## How to Run GUI (2 Options)
1. If the GUI executable has been installed, navigate to the appropriate location of the compliance_gui executable and run the file.
2. If the GUI executable has not been installed, open the compliance_gui.py file in pyCharm, Visual Studio Code, or another python IDE to run the file.

## How to Operate the GUI
1. Select the pressure data file and the video file.
2. If you wish, enter a specific pressure lower and upper bound to find the compliance. If the value exceeds 150 mmHg, errors may occur.
If not, the default will be 50-90 mmHg. 
3. Click Run and wait < 1 minute for the compliance outputs. 

## Important Notes about Performing Compliance Testing
1. To perform compliance testing, first make sure the bright rectangular LED lights  have sufficient charge. 
2. Fill glass tray with water and turn on water bath. Ensure the metal chassis is secured and fill with PBS. 
3. Insert light in zip ties and secure vessel with rubber band or suture. 
4. Position the vessel in frame using the pylon viewer and ensure the focus is on the edges. 
5. **It is very important to make sure there is as little debris floating in the metal chassis as possible.**
6. Set the exposure time to 40,000 (40 ms).
7. Run the test by starting the pressure recording, then starting the video recording at the same time or just before 
starting the syringe pump.
8. **It is very important to ensure the video and pressure are being saved onto the desktop for the timestamps to save correctly.**

## Important Note About Canny Edge Detection Threshold
The Canny Edge opencv function takes a lower and upper threshold that represent bounds for the gradient/first derivative of the image.
If the exposure time is set to 40,000 in pylon viewer, the bounds should be approximately 40-60. If the exposure time is adjusted to experiment with
removing debris in the image, the exposure time may need to be adjusted. For example, if the exposure time is 10,000, a threshold of 20-30 or 20-40 may be more appropriate.

## Potential Warnings or Errors

   A. If the video file does nto capture the entirety of the vessel being pressurized to 4 psi, some output values from the compliance code will be nan. The frames selected will essentially be the same
resulting in 0 difference in pixel diameter and a compliance of nan.

   B. If the Canny Filter Thresholds are too high, the image will potentially miss one edge and capture the next edge. If the left and right edges are deemed the same pixel, there will be a warning that there is math 
occurring with a 0 divisor. This is just a warning and does not influence the code. 

   C. If the video and pressure data are not saved to the desktop and then transferred via USB or over the shared drive, the timestamps will be off. This will 
result in the time stamp of the video file potentially being entirely before or after the pressure data. Error messages may occur and prevent the code from running. If this occurs, 
there is a commented code block in canny_edge.py to find the start of the video file based on when the 
pressure data begins to increase. This would need to be switched with the code line using the video timestamp. 

## How the Code Works

The code starts by taking a video file and pressure fata file input. Using timestamps from both, the data is aligned and 
pressure data cropped. The frames of the video are looped through and filtered using canny_edge detection filtering. The frames of interest are identified from when the 
pressure values of interest (ie: 50-90 mmHg) are found. The leftmost and rightmost non-zero points in every row are selected as the edges. The compliance is calculated in accordance with ISO 7198 
once the edges of the frames of interest are identified. The 

Outputs of the data include a visual indication of the calculated compliance on the GUI and an excel file output of the time, pressure, and compliance metrics saved in the 
folder with the video and pressure data files. 


## MIT License

Copyright (c) [2022] [Benjamin Alexander Randoing]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.