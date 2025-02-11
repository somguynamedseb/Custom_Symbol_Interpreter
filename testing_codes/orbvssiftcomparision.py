import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

MIN_MATCH_COUNT = 10
 
img1 = cv.imread('testqr.png', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('WIN_20240708_09_37_43_Pro.png', cv.IMREAD_GRAYSCALE) # trainImage
 

start = time.time()
# Create the ORB instance
orb = cv.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
 
# FLANN_INDEX_KDTREE = 1a
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,         # was 12
                   key_size = 12,            # was 20
                   multi_probe_level = 1)    # was 2
search_params = dict(checks = 50)
 
flann = cv.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

warped = None
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    outlinepts = np.float32([[0,0],[0,w],[h,w],[h,0]])
    M2 = cv.getPerspectiveTransform(dst,outlinepts)

    warped = cv.warpPerspective(img2, M2, (h,w)) 
    end = time.time()
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    
print("The time of execution of above program is :",(end-start) * 10**3, "ms")    
img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
plt.subplot(121),plt.imshow(img2),plt.title('Input')
plt.subplot(122),plt.imshow(warped),plt.title('Output')
plt.show()

 


