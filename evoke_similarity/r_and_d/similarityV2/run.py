# -*- coding: utf-8 -*-

from compareV2 import Compare
from many_extractorV2 import FeatureExtractor as fext
import numpy as np 
import pandas as pd 
from shutil import copyfile
from matplotlib import pyplot as plt
from PIL import Image 
import os
compOBJ = Compare() 
features = compOBJ.features
image_path_list = compOBJ.image_path_list
#single_image_path = '/home/mahdi/Pictures/test/8.png'
single_image_path = input('image path :  ')
fextOBJ = fext()
single_feature = fextOBJ.img2vector(single_image_path)
mse_all = []
ssim_all = [] 
dst_img_dir_mse = 'results/mse/'
dst_img_dir_ssim = 'results/ssim/'


for i in range(0,len(features)):
    feature = features[i]
    im_path = image_path_list[i]
    feature=np.array(feature)
    mse = compOBJ.mse_compare(single_feature,feature)
    ssim = compOBJ.ssim_compare(single_feature,feature)
    #cosin_all = compOBJ.cosin_compare(single_feature)
    mse_all.append([im_path,mse])
    ssim_all.append([im_path,ssim])

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