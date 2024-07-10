import cv2 as cv
from pyzbar.pyzbar import decode
import numpy as np

camera_id = 0
delay = 1
window_name = 'OpenCV pyzbar'

cap = cv.VideoCapture('drone qr test.mp4')

while True:
    ret, frame = cap.read()
    if ret:
        for d in decode(frame):
            s = d.data.decode()
            print(s)
            frame = cv.rectangle(frame, (d.rect.left, d.rect.top),
                                  (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (0, 255, 0), 3)
            frame = cv.putText(frame, s, (d.rect.left, d.rect.top + d.rect.height),
                                cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow(window_name, frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    

cv.destroyWindow(window_name)