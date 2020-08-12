# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2
import global_feature_extractor as fe
from skimage.measure import compare_ssim as ssim
#
#def mse(imageA, imageB):
#	# the 'Mean Squared Error' between the two images is the
#	# sum of the squared difference between the two images;
#	# NOTE: the two images must have the same dimension
#	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#	err /= float(imageA.shape[0] * imageA.shape[1])
#	
#	# return the MSE, the lower the error, the more "similar"
#	# the two images are
#	return err

file1 = '/home/mahdi/Pictures/test/1.jpeg'
file2 = '/home/mahdi/Pictures/test/3.jpeg'
#file1 = input('input first picture : ')
#file2 = input('input second picture : ')
fixed_size = tuple((500, 500))

fe_obj = fe.Global_feature_extraction()

####################################_______image1_______
image1 = cv2.imread(file1)
image1 = cv2.resize(image1, fixed_size)

####################################
# Global Feature extraction
####################################
shape1 = fe_obj.shape(image1)
texture1   = fe_obj.texture(image1)
color1  = fe_obj.color(image1)
####################################_______image2_______
image2 = cv2.imread(file2)
image2 = cv2.resize(image2, fixed_size)

####################################
# Global Feature extraction
####################################
shape2 = fe_obj.shape(image2)
texture2   = fe_obj.texture(image2) 
color2  = fe_obj.color(image2)
####################################


####################################
# collecting data on vector 
####################################

global_feature1 = np.hstack([color1, texture1, shape1])
global_feature2 = np.hstack([color2, texture2, shape2])




##################################
# MSE
##################################
mse = np.square(np.subtract(global_feature1,global_feature2)).mean()
#err = np.sum((global_feature1.astype("float64") - global_feature2.astype("float64")) ** 2)
print(mse)
ssim_compare = ssim(global_feature1,global_feature2)
