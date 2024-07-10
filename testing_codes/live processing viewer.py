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

initialdectector = cv.ORB_create(nfeatures = 20, scaleFactor = 1.2, nlevels = 2, edgeThreshold = 1, firstLevel = 0, WTA_K = 2, scoreType = cv.ORB_HARRIS_SCORE, patchSize = 10, fastThreshold = 100)
detector = cv.ORB_create(nfeatures = 200, scaleFactor = 1.5, nlevels = 4, edgeThreshold = 100, firstLevel = 0, WTA_K = 2, scoreType = cv.ORB_HARRIS_SCORE, patchSize = 10, fastThreshold = 100)

# detector = cv.SIFT_create()
# setting up parameters and orb/flann matcher instances
MIN_MATCH_COUNT = 7
orb_test = 'april tag.png'

img1 = cv.imread(orb_test, cv.IMREAD_GRAYSCALE) # queryImage ##TODO: hardcode this into a file for processing over head
ret,img1 = cv.threshold(img1,100,255,cv.THRESH_BINARY)
kp1, des1 = initialdectector.detectAndCompute(img1,None)
# kp1 = fast.detect(img1,None)
h0 = int(img1.shape[0])
w0 = int(img1.shape[1])
print(h0,w0)
img2 =  cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.namedWindow('img2',cv.WINDOW_NORMAL)
cv.resizeWindow('img2', w0,h0)

cv.imshow("img2", img2)

FLANN_INDEX_KDTREE =1 ##SIFT
FLANN_INDEX_LSH = 6 ##ORB
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,         # was 12
                   key_size = 10,            # was 20
                   multi_probe_level = 1)    # was 2
search_params = dict(checks = 100)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Create the sharpening kernel 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

# setup directories and video capture
vid = cv.VideoCapture('april test.mp4')
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
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = vid.read()  
    # Sharpen the image 
    try: ##put image processing in try block to catch errors and video ending easily
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        processed_frame = grey
        # processed_frame = cv.filter2D(processed_frame, -1, kernel) 
        # ret,processed_frame = cv.threshold(processed_frame,100,255,cv.THRESH_TOZERO)
        ret,processed_frame = cv.threshold(processed_frame,100,255,cv.THRESH_BINARY)
        
    except:
        print("done")
        break
        
    frame_count += 1
    kp2, des2 = detector.detectAndCompute(processed_frame,None)
    matches = flann.knnMatch(des1,des2,k=2)
    matches = [match for match in matches if len(match) == 2] ## sometimes knn match returns single point instead of the normal pair
    # cv.imshow('frame', frame)
    try:
        good = []
        for m,n in matches:
            # store all the good matches as per Lowe's ratio test.
            # print(m.distance, n.distance)
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
            # frame = cv.drawKeypoints(frame, kp2, None, color=(0,255,0), flags=0)
            frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
            # processed_frame = cv.drawKeypoints(processed_frame, kp2, None, color=(0,255,0), flags=0)
            # processed_frame = cv.polylines(processed_frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
            print(str(len(matches))+ "    "+ str(len(good)))
            frame = cv.drawKeypoints(frame, kp2, None, color=(0,255,0), flags=0)
            cv.imshow("frame", frame)
            # good = []
        else:   
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            # good = []
            pass
        print(str(len(matches))+ "    "+ str(len(good)))
    except:
        # cv.imshow("raw", frame)
        # good = []
        pass

    processed_frame = cv.drawKeypoints(processed_frame, kp2, None, color=(0,255,0), flags=0)
    
    cv.imshow("BWframe",processed_frame)

    
    

