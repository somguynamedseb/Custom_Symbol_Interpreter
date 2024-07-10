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




# setup directories and video capture
vid = cv.VideoCapture('red quad circle.mp4')
ret, frame = vid.read()



#initialize performance variables
frame_count = 0
frames_good = 0


while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = vid.read()  


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # lower mask (0-10)
    lower_red = np.array([0,75,75])
    upper_red = np.array([10,255,255])
    mask0 = cv.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,75,75])
    upper_red = np.array([180,255,255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    res = cv.bitwise_and(frame,frame, mask= mask)    

    # Blur using 3 * 3 kernel. 
    blurred = cv.blur(res, (7, 7)) 
    grey = cv.split(blurred)[2]
    # Apply Hough transform on the blurred image. 

    
    detected_circles = cv.HoughCircles(grey,  
                    cv.HOUGH_GRADIENT_ALT, 1, 20, param1 = 30, 
                param2 = .5, minRadius = 3, maxRadius = 20) 
    # print(len(detected_circles[0, :]))
    # Draw circles that are detected. 
    if detected_circles is not None: 
        
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            # Draw the circumference of the circle. 
            cv.circle(res, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv.circle(res, (a, b), 1, (0, 0, 255), 3) 
            cv.imshow("Detected Circle", res) 
            # cv.imshow("mask", mask)
