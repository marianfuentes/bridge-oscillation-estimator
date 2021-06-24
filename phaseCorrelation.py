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

#Select the video
#video_filename = 'out_printer0.5Hz.avi'
#video_filename = '1Hz-2.5mm.avi'
#video_filename = '10x,z.avi'
video_filename = '2.5x,z.avi'
#video_filename = ("final_5mm.AVI")
#video_filename = ("scope_9.AVI")

########functions##########
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem


# read the video on the selected path
camera = cv2.VideoCapture(video_filename)
# initialize parameters
frames_t = []
correlations = []
x = 0
y = 0
cont=0
corr_x=[]
corr_y=[]

ret, frame = camera.read()

resizex = frame.shape[0] // 2
resizey = frame.shape[1] // 2
frame1 = cv2.resize(frame, tuple([resizey, resizex]))
# Select object from image
r = cv2.selectROI(frame1);
# r = (top left column point, top left row point, distancte in columns from first point to second point, distance in rows to second point)
# ROI coordinates
upper_left = (int(r[1] * 2), int(r[0] * 2))
bottom_right = (int(r[3] * 2), int(r[2] * 2))

ROI=frame[upper_left[0] : upper_left[0]+bottom_right[0], upper_left[1] : upper_left[1]+bottom_right[1]]
cv2.imshow('ROI ', ROI )
cv2.waitKey(0)

ret = True
# loop in charges of process the video
while ret:

    ret, frame = camera.read()

    # verifies that the frame was correctly read
    if ret:
        # select the roi in the actual frame
        frames=frame[upper_left[0] : upper_left[0]+bottom_right[0], upper_left[1] : upper_left[1]+bottom_right[1]]
        # thresholds the actual roi
        ret, frames = cv2.threshold(frames, 65, 255, cv2.THRESH_BINARY)
        if cont < 1:
            # visualize the thresholding to verify that it shows only the objective
            cv2.imshow('Thersholding', frames)
            cv2.waitKey(0)
        # set up amplificacion value
        sizex = frame.shape[1]/frames.shape[1]
        sizey = frame.shape[0] / frames.shape[0]
        frames_t.append(frames)
        cont=cont+1

i=0
# set up the source 1 as the first frame, this frame its going to be our reference
src1 = frames_t[0]
src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
src1 = np.float32(src1)
time = 0
time_Total=[]
############### DISTANCE CALCULATION ################
# Vector that contains the camera parameters
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

axe_x = []
axe_y = []

# plot the movement

for i in range(0,len(frames_t)):


    src2=frames_t[i]
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    src2 = np.float32(src2)
    ret = cv2.phaseCorrelate(src1, src2)
    pixelesX=sizex*ret[0][0]
    pixelesY=sizey*ret[0][1]
    X1 = Zx/Fx * pixelesX
    Y1 = Zy/Fy * pixelesY
    position_x =  abs(X1)
    position_y =  abs(Y1)

    axe_x.append(position_x)
    axe_y.append(position_y)

    corr_x.append(X1)
    corr_y.append(Y1)
    actual_time=time*(1/30.0e-3)
    time_Total.append(actual_time)
    time=time+1


ytf=fft(axe_y)
N=len(axe_y)
print(N)
xtf=fftfreq(len(axe_y),(1/33.33))
ytf=ytf.real
ytf = ytf.tolist()
search_ytf=ytf[10:len(ytf)-1]
pos=search_ytf.index(max(search_ytf))
print(pos)
frecuencia=xtf[pos+10]
print('frecuencia calculada Phase Correrlation',frecuencia)

lvd_Data = pd.read_csv('distancia_scope_9.csv',sep=';')
fig, axs = plt.subplots(2)
axs[0].plot(lvd_Data['time'],lvd_Data['distance'])
axs[0].set_title('Lvdt distances')
axs[0].set(xlabel=" Time (s)",ylabel=" Distances (mm)")
axs[0].grid()
axs[1].plot(time_Total,axe_x,'r',label='X Distance')
axs[1].plot(time_Total,axe_y,'b',label='Y distance')
axs[1].set_title('Phase Correlation distances')
axs[1].set(xlabel=" Time (ms)",ylabel=" Distances (mm)")
axs[1].legend(loc='best', shadow=True)
axs[1].grid()
fig.tight_layout()
plt.show()
max_dis_Lvdt=max(lvd_Data['distance'])
max_dis_phase=max(axe_y)

print('Distancia Lvdt',max_dis_Lvdt)
print('Distancia Phase ',max_dis_phase)

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
axs[1].set_title('Phase Correlation FFT')
axs[1].set(xlabel=" Frequency (Hz)",ylabel=" |Power(f)|")
axs[1].grid()
fig.tight_layout()
plt.show()
## memory usage
mem_used=memory_usage_psutil()
print(mem_used)