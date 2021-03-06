# -*- coding: utf-8 -*-
"""
this module run similarity project and matching every part of our project together. 

NOTES :     
    
    - user should input image address.
    - pkl files should be generated by many_extractor madule before that you want to run.
    - create this object files : 
        - compare.py
        - many_extractor.py 
    - set the loop over the features to compare the single vector with all of data
    - create dataframe of mse and ssim list data 
    - sorting similar images on list
    - save similar images on results/mse and results/ssim directory.

Todo:
    * you should run this madule and input image addres that you wants to compare
"""
from compare import Compare
from many_extractor import FeatureExtractor as fext
import numpy as np 
from PIL import Image 
#create compare object
compOBJ = Compare() 
#import feature from compare objects
#you can find all of vector data that you need 
features = compOBJ.features
# import single image by user : you can use another input methods
single_image_path = input('image path :  ')
#create object of feature extractor on many_extractor file to make vector of single image.
fextOBJ = fext()
#import path lists for labeling the data and navigate them.
image_path_list = list(fextOBJ.data['image_path'])
single_feature = fextOBJ.img2vector(single_image_path)
#####################################################
mse_all = []
ssim_all = [] 
dst_img_dir_mse = 'results/mse/'
dst_img_dir_ssim = 'results/ssim/'

#loop over the features and image path list
for i in range(0,len(features)):
    feature = features[i]
    im_path = image_path_list[i]
    feature=np.array(feature)
    mse = compOBJ.mse_compare(single_feature,feature)
    ssim = compOBJ.ssim_compare(single_feature,feature)
    #cosin_all = compOBJ.cosin_compare(single_feature)
    mse_all.append([im_path,mse])
    ssim_all.append([im_path,ssim])

#create dataframe and give head of theme
df_mse , df_ssim = compOBJ.create_dataframe(mse_all,ssim_all)
head_mse , head_ssim = df_mse.head() , df_ssim.head()

def save_heads(head,swich):
    if swich == 'mse':
        for i in range(len(head)):
            img_path = head['image path'][i]
            img = Image.open(img_path)  
            img.save(dst_img_dir_mse+str(i)+'.jpg')
    else: 
        for i in range(len(head)):
            img_path = head['image path'][i]
            img = Image.open(img_path)  
            img.save(dst_img_dir_ssim+str(i)+'.jpg')
            
            
save_heads(head_mse,'mse')
save_heads(head_ssim,'ssim')