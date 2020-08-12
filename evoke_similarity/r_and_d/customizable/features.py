# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
#-----------------------------------
# GLOBAL FEATURE EXTRACTION
# LOCAL FEATURE EXTRACTION
#-----------------------------------

import mahotas
import cv2


class Global_feature_extraction:
    def __init__(self):
        self.bins = 8

    # feature-descriptor-1: Hu Moments
    def shape(self,image):
        """Example function with types documented in the docstring.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    # feature-descriptor-2: Haralick Texture
    def texture(self,image):
        """Example function with types documented in the docstring.
        """
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    # feature-descriptor-3: Color Histogram
    def color(self,image, mask=None):
        """Example function with types documented in the docstring.
        """
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()

    # def edge(self,img):
    #     """Example function with types documented in the docstring.
    #     """
    #     img = cv2.imread(img, 0)
    #     cv2.imwrite("canny.jpg", cv2.Canny(img, 200,300))

class Local_feature_extractor:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.surf = cv2.xfeatures2d.SURF_create(400)
        self.orb = cv2.ORB_create()
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.kaze = cv2.KAZE_create()
        #self.dense = cv2.FeatureDetector_create("Dense")
        
    def SIFT(self,img_path):
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
        img = cv2.imread(img_path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #sift = cv2.xfeatures2d.SIFT_create()
        #kp = self.sift.detect(gray,None)
        kp , des = self.sift.detectAndCompute(gray,None)
        #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
        #plt.imshow(img2),plt.show()        
        #cv2.imwrite('sift_keypoints.jpg',img2)
        cv2.normalize(des,des)
        
        return kp,des
        
        
        
    def SURF(self,img_path):
        """Example function with types documented in the docstring.
        """
        #its OK
        img = cv2.imread(img_path,0)
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        #surf = cv2.xfeatures2d.SURF_create(400)
        kp, des = self.surf.detectAndCompute(img,None)
        # Find keypoints and descriptors directly
        # We set it to some 50000. Remember, it is just for representing in picture.
        # In actual cases, it is better to have a value 300-500
#        surf.setHessianThreshold(50000)
#        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#        plt.imshow(img2),plt.show()
#        cv2.imwrite('surf_keypoints.jpg',img2)
        cv2.normalize(des,des)
        return kp,des

        
    def ORB(self,img_path):
        """Example function with types documented in the docstring.
        """
        img = cv2.imread(img_path,0)
        # Initiate STAR detector
        #orb = cv2.ORB_create()
        #find the keypoints with ORB and compute the descriptors with ORB
        kp, des = self.orb.detectAndCompute(img, None)
        # draw only keypoints location,not size and orientation
#        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#        plt.imshow(img2),plt.show()
#        cv2.imwrite('orb_keypoints.jpg',img2)
        # cv2.normalize(des,des)
        return kp,des

    def BRIEF(self,img_path):
        """Example function with types documented in the docstring.
        """
        img = cv2.imread(img_path,0)
        # Initiate STAR detector
        #star = cv2.xfeatures2d.StarDetector_create()
        # Initiate BRIEF extractor
        #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        # find the keypoints with STAR
        kp = self.star.detect(img,None)
        # compute the descriptors with BRIEF
        kp, des = self.brief.compute(img, kp)
#        img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#        plt.imshow(img2),plt.show()
#        cv2.imwrite('brief_keypoints.jpg',img2)
        cv2.normalize(des,des)
        return kp,des
        
    def KAZE(self,img_path):
        """Example function with types documented in the docstring.
        """
        img = cv2.imread(img_path,0)
        #alg = cv2.KAZE_create()
        kp = self.kaze.detect(img)
        kp, dsc = self.kaze.compute(img, kp)
        cv2.normalize(dsc,dsc)
        return kp,dsc
    

        
class ANN_feature_extractor:
    def __init__(self):
        pass
    def resnet(self):
        pass
    