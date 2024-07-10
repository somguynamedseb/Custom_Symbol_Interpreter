import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os
from datetime import date, datetime


# setup ORB PROCESSING and FLANN parameters
# Create the ORB instance

detector = cv.ORB_create(nfeatures = 10, scaleFactor = 1.1, nlevels = 8, edgeThreshold = 1, firstLevel = 0, WTA_K = 2, scoreType = cv.ORB_HARRIS_SCORE, patchSize = 5, fastThreshold = 20)
# detector = cv.SIFT_create()
# setting up parameters and orb/flann matcher instances
MIN_MATCH_COUNT = 10
orb_test = 'IDqr.png'

img1 = cv.imread(orb_test, cv.IMREAD_GRAYSCALE) # queryImage ##TODO: hardcode this into a file for processing over head
ret,img2 = cv.threshold(img1,100,255,cv.THRESH_BINARY_INV)
kp1, des1 = detector.detectAndCompute(img2,None)
# kp1 = fast.detect(img1,None)
h = int(img1.shape[0])
w = int(img1.shape[1])

cv.namedWindow('img2',cv.WINDOW_NORMAL)
cv.resizeWindow('img2', w,h)
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    img3 = cv.drawKeypoints(img2, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('img2', img3)

    

