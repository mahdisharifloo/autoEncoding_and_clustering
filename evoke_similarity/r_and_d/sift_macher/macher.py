

import matplotlib.pyplot as plt
import numpy as np
import cv2 
from local_feature_extractor import local_feature_extraction as lfe


image_path = '/home/mahdi/Pictures/nike.png'
image_path2 = '/home/mahdi/Pictures/nike_shoes.jpeg'

img1 = cv2.imread(image_path)
img2 = cv2.imread(image_path2)

lfe1 =lfe(image_path)
lfe2 =lfe(image_path2)


MIN_MATCH_COUNT = 10

surf_kp1,surf_des1 = lfe1.SURF()
surf_kp2,surf_des2 = lfe2.SURF()


#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(surf_des1,surf_des2)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))







# def extractFeatures_SIFT(): 
 
#     featurlist += [kp2, des2]
#     bf = cv2.BFMatcher()
#     matches1 = bf.knnMatch(des1,des2, k=2)
#     good = []
#     for m,n in matches1:
#         if m.distance < 0.7*n.distance:
#             good.append([m])
#             a=len(good)
#             print(a)
#             percent=(a*100)/kp2
#             print("{} % similarity".format(percent))
#             if percent >= 75.00:
#                 print('Match Found')
#                 break;
#     return featurlist
# print(featurlist)
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# if __name__ == '__main__':
#     extractFeatures_SIFT()