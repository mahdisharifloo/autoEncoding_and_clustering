# -*- coding: utf-8 -*-

#########################
# import libraries
#########################
import cv2
import numpy as np
from matplotlib import pyplot as plt

#########################
# operations
#########################
class local_feature_extraction:
    def __init__(self,img_path):
        self.img_path = img_path
    
    def SIFT(self):
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
        img = cv2.imread(self.img_path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()        
        cv2.imwrite('sift_keypoints.jpg',img2)
        return kp
        
        
        
    def SURF(self):
        #its OK
        img = cv2.imread(self.img_path,0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = surf.detectAndCompute(img,None)
        # Find keypoints and descriptors directly
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
        surf.setHessianThreshold(50000)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        cv2.imwrite('surf_keypoints.jpg',img2)
        return kp,des

        
    def ORB(self):
        img = cv2.imread(self.img_path,0)
        # Initiate STAR detector
        orb = cv2.ORB_create()
        #find the keypoints with ORB and compute the descriptors with ORB
        kp, des = orb.detectAndCompute(img, None)
        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        cv2.imwrite('orb_keypoints.jpg',img2)
        return kp,des

    def BRIEF(self):
        img = cv2.imread(self.img_path,0)
        # Initiate STAR detector
        star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = star.detect(img,None)
        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)
        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        plt.imshow(img2),plt.show()
        cv2.imwrite('brief_keypoints.jpg',img2)
        return kp,des
