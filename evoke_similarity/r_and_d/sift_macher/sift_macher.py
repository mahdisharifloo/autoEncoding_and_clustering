# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from local_feature_extractor import local_feature_extraction as lfe



image_path = '/home/mahdi/Pictures/1.jpeg'
image_path2 = '/home/mahdi/Pictures/nike_shoes.jpeg'

img1 = cv.imread(image_path)
img2 = cv.imread(image_path2)

lfe1 =lfe(image_path)
lfe2 =lfe(image_path2)


MIN_MATCH_COUNT = 10

surf_kp1,surf_des1 = lfe1.SURF()
surf_kp2,surf_des2 = lfe2.SURF()



FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(surf_des1,surf_des2,k=2)
# store all the good matches as per Lowe's ratio test.
#matches = sorted(matches, key=lambda val: val.distance)
good = []

# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)
          a=len(good)
          print(a)
          percent=(a/len(surf_kp2))*100
          print("{} % similarity".format(percent))
          if percent >= 75.00:
              print('Match Found')
              break;




if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ surf_kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ surf_kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,surf_kp1,img2,surf_kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()










#
#surf_kp,surf_des = lfe.SURF(img_path)
#orb_kp,orb_des =lfe.ORB(img_path)
#sift_kp = lfe.SIFT(img_path)
#brief_kp,brief_des = lfe.BRIEF(img_path)
#
#
#def macher(kp,des=None):
#    # create BFMatcher object
#    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#    # Match descriptors.
#    matches = bf.match(des1,des2)
#    # Sort them in the order of their distance.
#    matches = sorted(matches, key = lambda x:x.distance)
#    # Draw first 10 matches.
#    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    plt.imshow(img3),plt.show()
#    
#    
