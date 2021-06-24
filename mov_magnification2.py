import cv2
import numpy as np
import scipy.signal as signal
import time
import psutil
import os

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level):
    s=src.copy()
    pyramid=[s] #array to save pyramid levels
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
        #print("GAUSSIAN", s.shape)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        #print("SHAPE",GE.shape)
        #Resta la imagen gaussiana de un nivel más abajo de la imagen laplaciana actual
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    return pyramid


#load video from file
def load_video(video_filename):

    resize_factor = 2.5
    cap = cv2.VideoCapture(video_filename)

    mas = 9
    pixeles_mas = 5
    ###################### ROI SELECTOR ###################
    x = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            #Select ROI in the first frame
            if (x == 0):
                frame_copy = np.copy(frame)
                #print("frame_copy shape", frame_copy.shape)
                #If the frame is too big you need to resize
                #Resize frame in order to watch the whole image
                #Calculate resize parameters dividing original frame by a selected resize factor
                resize_row = frame_copy.shape[0] // resize_factor
                resize_col = frame_copy.shape[1] // resize_factor
                #Use the resize parameters tp create a resized frame "aux_frame"
                aux_frame = cv2.resize(frame_copy, tuple([int(resize_col), int(resize_row)]))
                #print("aux frame shape", aux_frame.shape)

                #Call the selectorROI function (draw a rectangle from top left and drag to bottom right)
                r = cv2.selectROI(aux_frame)

                #OJO! r returns:
                #r = (top left column point, top left row point, distancte in columns from first point to second point, distance in rows to second point)
                #print("r",r)
            x += 1
            # else:
            # break
        else:
            break

############################### END OF SELECTOR ###################################################

    #rectangle points of ROI (Region of Interest) trasnformed to equivalent points in original image
    upper_left = (round(r[1]*resize_factor), round(r[0]*resize_factor))
    bottom_right = (round(r[3]*resize_factor), round(r[2]*resize_factor))

    #upper_left = (838, 765)
    #bottom_right = (435, 530)

    #upper_left = (320, 648)
    #bottom_right = (358, 688)

    print('empieza en', upper_left)
    print('va hasta', bottom_right)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #print(fps, height, width, frame_count)

    # Calculate height and width of ROI image
    h = abs((upper_left[0] - (mas * pixeles_mas)) - ((upper_left[0]+bottom_right[0])+ (mas * pixeles_mas)) )
    w = abs((upper_left[1]- (mas * pixeles_mas) ) - ((upper_left[1]+bottom_right[1])+ (mas * pixeles_mas)) )
    print("HEIGHT", h, "WIDTH", w)

    ## Adjustment for ROI size, it must be divisible by 8 because of the 3 pyramid levels (2^3 = 8)
    fixed = False
    fix_h = 0
    fix_w = 0
    fixed_h = False
    fixed_w = False

    #When it detects that ROI height and width  is NOT divisible by 8, it adds a pixel to each side until it becomes divisible.
    while(fixed == False):

        if (h % 8 == 0):
            fixed_h = True
        else:
            fix_h = fix_h +1
            h = h + 1

        if (w % 8 == 0):
            fixed_w = True
        else:
            fix_w = fix_w + 1
            w = w + 1

        if((fixed_w and fixed_h) == True):
            fixed = True

    print("HEIGHT 2", h, "WIDTH 2", w)
    #Create tensor (array) of width columns, 3 rows, and [fram_count][height] frames
    #básicamente es un vector de frames vacíos que se va llenando con los frames

    #Construct a zero matriz of shape frames, h, w, 3
    video_tensor=np.zeros((frame_count,h,w,3),dtype='float')
    print("vidtensor shape", video_tensor.shape)


    ######### SAVE ROI FRAMES IN VIDEO TENSOR MATRIZ ##################

    cap = cv2.VideoCapture(video_filename)
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame[(upper_left[0]-(mas * pixeles_mas)) :upper_left[0]+bottom_right[0]+(mas * pixeles_mas) + fix_h, upper_left[1] - (mas * pixeles_mas) :upper_left[1]+bottom_right[1]+ (mas * pixeles_mas)+fix_w]
            if(x==0):
                f = frame[(upper_left[0]-(mas * pixeles_mas)) :upper_left[0]+bottom_right[0]+(mas * pixeles_mas) + fix_h, upper_left[1] - (mas * pixeles_mas) :upper_left[1]+bottom_right[1]+ (mas * pixeles_mas)+fix_w]
                print("ROI size",f.shape)
                cv2.imshow('ROI new',frame[(upper_left[0]-(mas * pixeles_mas)) :upper_left[0]+bottom_right[0]+(mas * pixeles_mas) + fix_h, upper_left[1] - (mas * pixeles_mas) :upper_left[1]+bottom_right[1]+ (mas * pixeles_mas)+fix_w])
                cv2.waitKey(0)
            x+=1
        else:
            break
    return video_tensor,fps

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels):
    tensor_list=[]

    #recorre el número de frames del video
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,levels=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

#reconstruct video from laplacian pyramid
def reconstruct_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
        final[i]=up
    return final

#save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("0.8Hz-2.5mmMAGMAG2.avi", fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()


########### MAIN FUNCTION ####################
if __name__=="__main__":
    path = "C:/Users/MarianEstefaniaFuent/Documents/Práctica/AAAAMARIAN/"
    image_name = '0.8Hz-2.5mm.avi' #Alejo_2.mp4 0.8Hz-2.5mm.avi printer1.mp4
    path_file = os.path.join(path, image_name)

    #t, f = load_video(path_file)

    ###################################################################
    ####### Parameters that show good results for skin color amp ######
    ###################################################################
    levels = 3# Number of gaussian pyramid levels constructed for every frame
    amplification = 20
    low_f = 0.5
    high_f = 1.2
    ###################################################################
    ############## Apply Color EVM and measure times ##################
    ###################################################################

    #start = time.perf_counter()  # Measuring load video time in s
    t, f = load_video(path_file)
    #stop = time.perf_counter()  # Measuring filter implementation time in s
    #load_video_time = stop - start

    start = time.perf_counter()  # Measuring gaussian video time in s
    lap_video_list = laplacian_video(t, levels=levels)
    filter_tensor_list = []
    #stop = time.perf_counter()  # Measuring gaussian time in s
    #gauss_video_time = stop - start

    #start = time.perf_counter()  # Measuring filter implementation time in s

    for i in range(levels):
        filter_tensor = butter_bandpass_filter(lap_video_list[i], low_f, high_f, f)
        filter_tensor *= amplification
        filter_tensor_list.append(filter_tensor)

    #filter_time = stop - start

    #start = time.perf_counter()  # Measuring reconstruction time in s
    recon = reconstruct_from_tensorlist(filter_tensor_list)
    #stop = time.perf_counter()  # Measuring reconstruction time in s
    #video_time = stop - start
    final_video = t + recon  # Measuring filter implementation time in s
    save_video(final_video)
    stop = time.perf_counter()

    total_time = stop - start

    process = psutil.Process(os.getpid())
    mem2 = process.memory_info().rss / float(2 ** 20)
    #print('loading video time is: ', load_video_time)
    #rint('gaussian video time construction is: ', gauss_video_time)
    #print('filter implementation time is: ', filter_time)
    #print('video reconstruction time is: ', video_time)
    print("Total magnification processing time is ", total_time)
    print("Memory: ", mem2)