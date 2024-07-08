import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 10
 
img1 = cv.imread('testqr.png', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('WIN_20240708_09_37_43_Pro.png', cv.IMREAD_GRAYSCALE) # trainImage
 
# Initiate SIFT detector
sift = cv.SIFT_create()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

warped = img2.copy()
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
    #M is 3x3 matrix
    #mask is 
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    outlinepts = np.float32([[0,0],[0,700],[700,700],[700,0]])
    M2 = cv.getPerspectiveTransform(dst,outlinepts)


    
    warped = cv.warpPerspective(img2, M2, (700,700))
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

 
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    
# warped =    
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
 singlePointColor = None,
 matchesMask = matchesMask, # draw only inliers
 flags = 2)
 
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
img3  = cv.drawKeypoints(img1, kp1, img2)
plt.subplot(121),plt.imshow(img3),plt.title('Input')
plt.subplot(122),plt.imshow(warped),plt.title('Output')

plt.show()