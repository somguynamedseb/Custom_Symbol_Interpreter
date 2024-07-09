import cv2 as cv 
import numpy as np

dir = "drone qr test.MP4"
vid = cv.VideoCapture(dir)
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # BW_Frame = cv.adaptiveThreshold(frame,100,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,5,2)
    # blur = cv.GaussianBlur(frame,(3,3),0)
    # ret3,BW_Frame = cv.threshold(frame,100,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret,thresh4 = cv.threshold(frame,120,255,cv.THRESH_TOZERO)
    frame = cv.filter2D(frame, -1, kernel)
    # closing = cv.morphologyEx(BW_Frame, cv.MORPH_OPEN, rect_kernel)
    cv.imshow('frame', thresh4)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


