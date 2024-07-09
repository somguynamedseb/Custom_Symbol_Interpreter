import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os
from datetime import date, datetime


notes = "testing to zero theshing and sharpening"


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()




# setup ORB PROCESSING and FLANN parameters
# Create the ORB instance
orb = cv.ORB_create()

# setting up parameters and orb/flann matcher instances
MIN_MATCH_COUNT = 10
orb_test = 'IDqr.png'

img1 = cv.imread(orb_test, cv.IMREAD_GRAYSCALE) # queryImage ##TODO: hardcode this into a file for processing over head
kp1, des1 = orb.detectAndCompute(img1,None)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,         # was 12
                   key_size = 12,            # was 20
                   multi_probe_level = 1)    # was 2
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

# setup directories and video capture
vid = cv.VideoCapture('drone qr test.mp4')
ret, frame = vid.read()
h = int(frame.shape[0]/2)
w = int(frame.shape[1]/2)

cv.namedWindow('frame',cv.WINDOW_NORMAL)
cv.resizeWindow('frame', w,h)
cv.namedWindow('BWframe',cv.WINDOW_NORMAL)
cv.resizeWindow('BWframe', w,h)


#initialize performance variables
frame_count = 0
frames_good = 0
while True:

    ret, frame = vid.read()  
    printProgressBar(frame_count, 1301, prefix = 'Progress:', suffix = 'Complete', length = 50)  
    # Sharpen the image 
    try: ##put image processing in try block to catch errors and video ending easily
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        processed_frame = grey
        # processed_frame = cv.filter2D(processed_frame, -1, kernel) 
        # ret,processed_frame = cv.threshold(processed_frame,140,255,cv.THRESH_TOZERO)
    except:
        print("done")
        break
        
    frame_count += 1
    kp2, des2 = orb.detectAndCompute(processed_frame,None)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # cv.imshow('frame', frame)
    try:
        goodq = []
        for m,n in matches:
            # store all the good matches as per Lowe's ratio test.
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>=MIN_MATCH_COUNT:
            frames_good += 1
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            outlinepts = np.float32([[0,0],[0,w],[h,w],[h,0]])
            M2 = cv.getPerspectiveTransform(dst,outlinepts)
            warped = cv.warpPerspective(frame, M2, (h,w)) 
            
            end = time.time()
            # cv.imshow('qr', warped)
            frame = cv.drawKeypoints(frame, kp2, None, color=(0,255,0), flags=0)
            frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
            processed_frame = cv.drawKeypoints(processed_frame, kp2, None, color=(0,255,0), flags=0)
            processed_frame = cv.polylines(processed_frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
            
            good = []
        else:   
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            good = []
            
    except:
        # cv.imshow("raw", frame)
        good = []
    cv.imshow("BWframe",processed_frame)
    cv.imshow("frame", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

