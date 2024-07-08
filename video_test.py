import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

# setup ORB PROCESSING and FLANN parameters
# Create the ORB instance
orb = cv.ORB_create()

MIN_MATCH_COUNT = 10
img1 = cv.imread('testqr.png', cv.IMREAD_GRAYSCALE) # queryImage ##TODO: hardcode this into a file for processing over head
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
vid = cv.VideoCapture(0)
ret, frame = vid.read()
while True:

    start= time.time()
    ret, frame = vid.read()

    
    # Sharpen the image 

    frame = cv.filter2D(frame, -1, kernel) 
    kp2, des2 = orb.detectAndCompute(frame,None)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    # cv.imshow('frame', frame)
    try:
        goodq = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>=10:
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
            cv.imshow('qr', warped)
            frame = cv.drawKeypoints(frame, kp2, None, color=(0,255,0), flags=0)
            frame = cv.polylines(frame,[np.int32(dst)],True,255,3, cv.LINE_AA)
            
            print("processing time = ", (end-start) * 10**3, "ms")
            cv.imshow("raw", frame)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            good = []
            
    except:
        cv.imshow("raw", frame)
        good = []
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break