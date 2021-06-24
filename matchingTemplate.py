########### Proyecto de Grado I ######################
########### Marian Fuentes - Alejandro Morales #######

import math as mt
import numpy as np
import cv2 as cv2
import pandas as pd
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
import time
import psutil
import os


########functions##########
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

#Select the video

#path = ("10x,z.AVI")
path = ("2.5x,z.AVI")
#path = ("2Hz-2.5mm.AVI")
#path = ("0.8Hz-2.5mm.AVI")
#path = ("scope_7_mag.avi")
#path = ("final_5mm.AVI")
#path = ("scope_12.AVI")
#path= ("0.8Hz-2.5mm_mag2.avi")
#path = ("5Hz-2.5mm.AVI")
#path = ("1Hz-2.5mm.AVI")
#path = ("0.5Hz-2.5mm.AVI")
#path = ("0.8Hz-2.5mm.AVI")
#path = ("5,x,2.5z.avi")

# read the video on the selected path
camera = cv2.VideoCapture(path)

ret, frame = camera.read()

resizex = frame.shape[0] // 2
resizey = frame.shape[1] // 2
frame1 = cv2.resize(frame, tuple([resizey, resizex]))
r = cv2.selectROI(frame1);
# Select object from image
# r = (top left column point, top left row point, distancte in columns from first point to second point, distance in rows to second point)
# ROI coordinates
upper_left = (int(r[1] * 2), int(r[0] * 2))
bottom_right = (int(r[3] * 2), int(r[2] * 2))

template=frame[upper_left[0] : upper_left[0]+bottom_right[0], upper_left[1] : upper_left[1]+bottom_right[1]]
cv2.imshow('ROI ', template )
cv2.waitKey(0)

# initialization
time_frame=0
time_video=[]
Euclidean_distances = []
distances=[]
video_time=[]

# number of pixels that the comparative image will be bigger (for each side)

pixels= 40

# Interpolation parameters to amplify the template
amp = 3
interpolated_Template = cv2.resize(src=template, dsize=None, fx=amp, fy=amp, interpolation=cv2.INTER_LANCZOS4)
#cv2.imshow('Interpolation', interpolated_Template)
#cv2.waitKey(0)
ret = True
# loop in charges of process the video
while ret:

    ret, frame = camera.read()

    # verifies that the frame was correctly read
    if ret:
        ## start time counter
        start = time.perf_counter()
        ## creates the frame compared to the previously selected points
        compared_frame = frame[(upper_left[0] - pixels) : (upper_left[0]+bottom_right[0] + pixels), (upper_left[1] - pixels ) : (upper_left[1]+bottom_right[1] + pixels )]
        ## INTERPOLATION(LANCZOS)
        # Interpolation parameters to amplify the compared frame
        interpolated_Compared = cv2.resize(src=compared_frame, dsize=None, fx=amp, fy=amp, interpolation=cv2.INTER_LANCZOS4)
        # CORRELATION
        # Function that generate a correlation pixel by pixel between the interpolated frame and interpolated template
        corr = cv2.matchTemplate(interpolated_Compared, interpolated_Template, cv2.TM_CCORR)
        ## GAUSSIAN FILTER, WINDOW EQUAL TO THE INTERPOLATION (3X3)
        # This filter helps to smooth the image, for a better search of the maximum correlation point
        smooth_corr = cv2.GaussianBlur(corr, (amp,amp), 0)
        # this plots are in charges of visualize the results
        #plt.subplot(221), plt.imshow(corr, cmap='gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.subplot(222), plt.imshow(smooth_corr, cmap='gray')
        #plt.title('Matching Result blurred'), plt.xticks([]), plt.yticks([])
        #plt.subplot(223), plt.imshow(cv2.cvtColor(interpolated_Compared, cv2.COLOR_BGR2RGB))
        #plt.title('Compared frame Interpolated'), plt.xticks([]), plt.yticks([])
        #plt.subplot(224), plt.imshow(cv2.cvtColor(interpolated_Template, cv2.COLOR_BGR2RGB))
        #plt.title('Template Interpolated'), plt.xticks([]), plt.yticks([])

        #plt.suptitle("Correlation")
        #plt.show()

        # MIN MAX SCALER
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(smooth_corr)

        print("Mínimo:", min_val, "Localizado en:", min_loc, "Máximo:", max_val, "Localizado en:", max_loc)

        ############### DISTANCE CALCULATION ################
        # Vector that contains the camera parameters
        K = [[
            3.74640869e+03,
            0.0,
            4.91808908e+02
        ],
            [
                0.0,
                0.91832865e+04,
                4.90443331e+02
            ],
            [0.0, 0.0, 1.0]]

        # Rescale the points to the original size

        StartingPoint = [((upper_left[0] + (bottom_right[0] // 2))), ((upper_left[1]+(bottom_right[1] // 2)))]

        YonPixels= (((upper_left[1]+(bottom_right[1] // 2))) - (smooth_corr.shape[0]//2)+max_loc[1])
        XonPixels= (((upper_left[0] + (bottom_right[0] // 2))) - (smooth_corr.shape[1]//2)+max_loc[0])
        #print(YonPixels)

        ActualPoint = [XonPixels, YonPixels]

        yy1 = StartingPoint[1]
        xx1 = StartingPoint[0]

        yy2 = ActualPoint[1]
        xx2 = ActualPoint[0]

        # Intrinsic parameters of the camera
        Fx = K[0][0]
        Fy = K[1][1]
        Cx = K[0][2]
        Cy = K[1][2]

        # Zx and Zy corresponds to the distance of the camera in millimeters
        Zx = 1070
        Zy = 1070
        # equations that convert the pixel coordinate to millimeters
        X1 = Zx * ((xx1 - Cx) / Fx)
        Y1 = Zy * ((yy1 - Cy) / Fy)

        X2 = Zx * ((xx2 - Cx) / Fx)
        Y2 = Zy * ((yy2 - Cy) / Fy)

        distanceX = abs(X2 - X1)
        distanceY = abs(Y2 - Y1)

        #print('distancia en X',distanceX)

        # the distances will be stored in "distance" vector
        Euclidean_distance=mt.sqrt((distanceX ** 2 )+(distanceY ** 2))
        distance=[distanceX,distanceY]

        distances.append(distance)
        Euclidean_distances.append(Euclidean_distance)
        actualTime = (1/30.0e-3) * time_frame
        time_video.append(actualTime)
        time_frame=time_frame + 1
        stop = time.perf_counter()
        video_time_act = stop - start
        video_time.append(video_time_act)


# plot the movement
mean_time=sum(video_time)/len(video_time)
print(mean_time)
Total_time=sum(video_time)
x_d=[]
y_d=[]
time_use=[]
freq=[]
i=0
a=0
Fs=1/30.0
for i in range(0,time_frame-1):
    x_d.append(distances[i][0])
    y_d.append(distances[i][1])
    time_use.append(time_video[i])

ytf=fft(y_d)
N=len(y_d)
print(N)
xtf=fftfreq(len(y_d),(1/33.33))
ytf=ytf.real
ytf = ytf.tolist()
search_ytf=ytf[10:len(ytf)-1]
pos=search_ytf.index(max(search_ytf))
print(pos)
frecuencia=xtf[pos+10]
print('frecuencia calculada Matching',frecuencia)

lvd_Data = pd.read_csv('distancia_scope_7.csv',sep=';')
fig, axs = plt.subplots(2)
axs[0].plot(lvd_Data['time'],lvd_Data['distance'])
axs[0].set_title('Lvdt distances')
axs[0].set(xlabel=" Time (s)",ylabel=" Distances (mm)")
axs[0].grid()
axs[1].plot(time_use,x_d,'r',label='X Distance')
axs[1].plot(time_use,y_d,'b',label='Y distance')
axs[1].set_title('Matching Template distances')
axs[1].set(xlabel=" Time (ms)",ylabel=" Distances (mm)")
axs[1].legend(loc='best', shadow=True)
axs[1].grid()
fig.tight_layout()
plt.show()
max_dis_Lvdt=max(lvd_Data['distance'])
max_dis_Matching=max(y_d)

print('Distancia Lvdt',max_dis_Lvdt)
print('Distancia Matching Template ',max_dis_Matching)

distances_lvdt=lvd_Data['distance'].to_numpy()
ytf1=fft(distances_lvdt)
N1=len(distances_lvdt)
print(N1)
xtf1=fftfreq(len(distances_lvdt),(0.0001))
ytf1=ytf1.real
ytf1 = ytf1.tolist()
search_ytf1=ytf1[10:N1//500]
pos1=search_ytf1.index(max(search_ytf1))
frecuencia=xtf1[pos1+10]
print('frecuencia calculada Lvdt',frecuencia)

fig, axs = plt.subplots(2)
axs[0].semilogy(xtf1[0:N1//500], 2.0/N1 * np.abs(ytf1[0:N1//500]))
axs[0].set_title('Lvdt FFT')
axs[0].set(xlabel=" Frequency (Hz)",ylabel=" |Power(f)|")
axs[0].grid()
axs[1].semilogy(xtf[0:N//2], 2.0/N * np.abs(ytf[0:N//2]))
axs[1].set_title('Matching Template FFT')
axs[1].set(xlabel=" Frequency (Hz)",ylabel=" |Power(f)|")
axs[1].grid()
fig.tight_layout()
plt.show()
mem_used=memory_usage_psutil()
print(mem_used)