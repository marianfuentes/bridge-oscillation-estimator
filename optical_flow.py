#################### TRABAJO DE GRADO II ###################
########### PONTIFICIA UNIVERSIDAD JAVERIANA ###############
####### MARIAN FUENTES Y ALEJANDRO MORALES #################

# DESCPRIPTION
# This code reads a video, ask the user to select a ROI with the mouse and to press INTRO twice in order to continue.
# It also reads a csv file with LVDT data to compare it with algorithm results.
# Then, it calculates dense optical flow within the selected ROI and plots it, along with LVDT displacement.

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import time
# return the memory usage in MB
import psutil
import os

################# VARIABLES ################
# Displacement for x and y for each iteration
dx = 0
dy = 0


#Array of displacements, collects displacements for x and y (variables dx and dy respectively)
axe_x = []
axe_y = []

#video time variables
ctime = 0
time_Total=[]

#Camera calibration matrix and variables
K = [[
            3.74640869e+03,
            0.0,
            4.91808908e+02
        ],
            [
                0.0,
                1.70832865e+04,
                4.90443331e+02
            ],
            [0.0, 0.0, 1.0]]

Fx = K[0][0]
Fy = K[1][1]
Cx = K[0][2]
Cy = K[1][2]
Zx=1070
Zy=1070

######### END OF MAIN VARIABLES #############


###### LOAD MAGNIFIED VIDEO ##############
video_filename = ("scope_8_trim.mp4")
cap = cv.VideoCapture(video_filename)
camera = cv.VideoCapture(video_filename)
ret, frame = camera.read()

######### READ LVDT DATA ###########
lvd_Data = pd.read_csv('distancia_scope_8.csv',sep=';')

############# RESIZE VIDEO ##############
#If the video size is bigger than you screen size, it's necessary to resize in order to appreciate the complete image.
resize_factor = 2

camera = cv.VideoCapture(video_filename)
ret, frame = camera.read()
frame_copy = np.copy(frame)

resize_row = frame_copy.shape[0] // resize_factor
resize_col = frame_copy.shape[1] // resize_factor

aux_frame = cv.resize(frame_copy, tuple([int(resize_col), int(resize_row)]))


######### SELECT ROI ####################

print("Please, click on the video and draw the desired Region of Interest.")
print("When you feel happy with your Region of Interest, press INTRO twice (two times) in order to continue")
#r = (top left column point, top left row point, distancte in columns from first point to second point, distance in rows to second point)
#r = cv.selectROI(frame_copy)
r = cv.selectROI(aux_frame)
print("Wait for the video to run and the distance plot to appear")

#ROI coordinates
#The resized image coordinates need to be changed to original frame coordinates.
upper_left_corner = (round(r[1]*resize_factor), round(r[0]*resize_factor)) #Save the upper left corner coordinates of the ROI
roi_sides = (round(r[3]*resize_factor), round(r[2]*resize_factor)) #Save the height and width of ROI

##### SAVE FIRST FRAME ##########
#Select the ROI in the original frame and save it as first_frame

add_pixels_flag = 0 #boolean to activate ROI size change iteration
pixels_added = 5 #Each iteration adds 5 pixels to height and width to ROI size

#Show ROI selected
first_frame=frame[(upper_left_corner[0]-(add_pixels_flag*pixels_added)) : upper_left_corner[0]+roi_sides[0]+(add_pixels_flag*pixels_added), upper_left_corner[1]-(add_pixels_flag*pixels_added) : upper_left_corner[1]+roi_sides[1]+(add_pixels_flag*pixels_added)]
cv.imshow('ROI new', first_frame)
cv.waitKey(0)

#Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
#Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
#Sets image saturation to maximum
mask[..., 1] = 255

processing_time_values = []
## Calculate memory used
process = psutil.Process(os.getpid())
mem = process.memory_info().rss / float(2 ** 20)
#print("memory used:", mem)

####### OPEN VIDEO AND CALCULATE DENSE OPTICAL FLOW  ###########
while (cap.isOpened()):

    ret, frame = cap.read()

    #breaks "while" loop when video reading ends
    if ret == False:
        break

    #Select frame according to the ROI
    frame = frame[(upper_left_corner[0]-(add_pixels_flag*pixels_added)) : upper_left_corner[0]+roi_sides[0]+(add_pixels_flag*pixels_added), upper_left_corner[1]-(add_pixels_flag*pixels_added) : upper_left_corner[1]+roi_sides[1]+(add_pixels_flag*pixels_added)]

    # Opens a new window and displays the input frame
    #cv.imshow("input", frame)

    start = time.perf_counter()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculates dense optical flow by Farneback method, compares previous gray scale frame with the present gray scale frame
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 5, 250, 5, 7, 1.2, 0)

    #Calculates max value of displacement for x and y inside the ROI

    x_flow_significant_value = [] #For each iteration save the values resulting from filtering optical flow values
    x_flow_significant_values = [] #Save all the signifcant optical flow values to perform later mean operation.

    y_flow_significant_value = []
    y_flow_significant_values = []

    # Calculates mean value of displacement for x and y inside the ROI filtering between boundaries specified b
    flowx = flow[...,0].copy() #Optical flow for displacement in x
    flowy = flow[...,1].copy() #Optical flow for displacement in y

    ## OPTICAL FLOW VALUES FILTER ##

    filter_num = 1 #Set this value according to the optical flow results in order to
    for y in flowy:
        y_flow_significant_value = y[y>filter_num]
        if len(y_flow_significant_value) == 0:
            fy = True
        else:
            y_flow_significant_values = np.concatenate((y_flow_significant_values, y_flow_significant_value),axis=0)
            #print('a', y_flow_significant_values)

        y_flow_significant_value = y[y < -filter_num]
        if len(y_flow_significant_value) == 0:
            fy = False
        else:
            y_flow_significant_values = np.concatenate((y_flow_significant_values, y_flow_significant_value),axis=0)

    for x in flowx:
        x_flow_significant_value = x[x>filter_num]
        #print('c',x_flow_significant_value)
        if len(x_flow_significant_value) == 0:
            fx = True
        else:
            x_flow_significant_values= np.concatenate((x_flow_significant_values, x_flow_significant_value),axis=0)
            #print('d', x_flow_significant_values)

        x_flow_significant_value = x[x < -filter_num]
        if len(x_flow_significant_value) == 0:
            fx = True
        else:
            x_flow_significant_values = np.concatenate((x_flow_significant_values, x_flow_significant_value),axis=0)


    #print(x_flow_significant_values.size,y_flow_significant_values.size)

    # Save displacements in array for each x and y

    if len(x_flow_significant_values) == 0:
        cerox = True
    else:
         cerox = False
         #print("ssss", x_flow_significant_values, "pppp", len(x_flow_significant_values))
         dx = sum(x_flow_significant_values)/len(x_flow_significant_values)

    if len(y_flow_significant_values) == 0:
        ceroy = True
    else:
        ceroy = False
        #print("sssaaaaaas", y_flow_significant_values, "ppaaapp", len(y_flow_significant_values))
        dy = sum(y_flow_significant_values)/len(y_flow_significant_values)


    #Convert pixels to mm

    #constant factor according to camera calibration
    x_camera_factor = Zx / Fx
    y_camera_factor = Zy / Fy

    dx = (dx * x_camera_factor) #multiply for and adjustment factor
    dy = (dy * y_camera_factor) #multiply for and adjustment factor

    axe_y.append(dy)
    axe_x.append(dx)

    stop = time.perf_counter()
    processing_time = stop - start

    processing_time_values.append(processing_time)

    # Computes the magnitude and angle of the 2D vectors
    #Change displacement for x and y into polar coordinates
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])


    #print("x flow", flow[...,0], "y flow", flow[...,1])
    #print('mag', magnitude, 'angle', angle)

    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    #cv.imshow("dense optical flow", rgb)

    #Update previous frame.
    #!!!! ATTENTION !!! only if you want to compare between consecutive frames, please update the frame
    prev_gray = gray


    ## Time video calculation, sample time = 30 ms
    actual_time = ctime * (1 / 30.0e-3)
    time_Total.append(actual_time)
    ctime = ctime + 1

    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

ROI_height = abs((  upper_left_corner[0] - (add_pixels_flag * pixels_added )  )-( upper_left_corner[0] + roi_sides[0] + (add_pixels_flag * pixels_added)) )
ROI_width = abs(( upper_left_corner[1] - (add_pixels_flag* pixels_added) ) - ( upper_left_corner[1] + roi_sides[1] + (add_pixels_flag * pixels_added) ))

    # The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()

total_processing_time = sum(processing_time_values)
average_processing_time_per_iteration = sum(processing_time_values)/len(processing_time_values)

############## PLOT LVDT distances and Dense optical flow distances ###########

# plt.subplots(121)
# plt.title("cero")
# plt.hist(flow[...,0], bins=50)
# plt.show()
# plt.subplot(122)
# plt.title("uno")
# plt.hist(flow[...,1], bins=50)
# plt.show()

fig, axs = plt.subplots(2)

##### LVDT DATA PLOT ######################
axs[0].plot(lvd_Data['time'],lvd_Data['distance'])
axs[0].set_title('LVDT data')
axs[0].set(xlabel=" Time (s)",ylabel=" Distance (mm)")
axs[0].grid()

######## DENSE OPTICAL FLOW PLOT ################
axs[1].plot(time_Total,axe_x,'r',label='X axe')
axs[1].plot(time_Total,axe_y,'b',label='Y axe')
axs[1].set_title('Optical flow data')
axs[1].set(xlabel=" Time (ms)",ylabel=" Velocity (mm/s)")
axs[1].legend(loc='best', shadow=True)
axs[1].grid()
fig.tight_layout()
plt.show()


########### PLOT FFT ##################

ytf = []
xtf = []

distances_lvdt=lvd_Data['distance'].to_numpy()
ytf1=fft(distances_lvdt)
N1=len(distances_lvdt)
#print(N1)
xtf1=fftfreq(len(distances_lvdt),(0.0001))
ytf1=ytf1.real
ytf1 = ytf1.tolist()
search_ytf1=ytf1[10:N1//500]
pos1=search_ytf1.index(max(search_ytf1))
frecuencia1=xtf1[pos1+10]


N=len(axe_y)
ytf=fft(axe_y)
xtf=fftfreq(len(axe_y),(1/33.33))
ytf=ytf.real
ytf = ytf.tolist()
search_ytf=ytf[10:len(ytf)-1]
pos=search_ytf.index(max(search_ytf))
frecuencia2=xtf[pos+10]


fig, axs = plt.subplots(2)
axs[0].semilogy(xtf1[0:N1//500], 2.0/N1 * np.abs(ytf1[0:N1//500]))
axs[0].set_title('Lvdt FFT')
axs[0].set(xlabel=" Frequency (Hz)",ylabel=" |Power(f)|")
axs[0].grid()
axs[1].semilogy(xtf[0:N//2], 2.0/N * np.abs(ytf[0:N//2]))
axs[1].set_title('Dense optical flow FFT')
axs[1].set(xlabel=" Frequency (Hz)",ylabel=" |Power(f)|")
axs[1].grid()
fig.tight_layout()
plt.show()


process = psutil.Process(os.getpid())
mem2 = process.memory_info().rss / float(2 ** 20)
#np.savetxt("datatesis.csv",mem, delimiter=';')

#np.savetxt("datatesis.csv",frecuencia1, delimiter=';')



print("Processing total time ", total_processing_time)
print("Average time per iteration ", average_processing_time_per_iteration)
print('Lvdt frequency',frecuencia1)
print('Optical flow frequency',abs(frecuencia2))
print("Height", ROI_height, "Width", ROI_width)
print("Memory used: ", mem2)

