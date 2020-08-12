# -*- coding: utf-8 -*-
"""
this module compairing twe image feature vectors.

feature vectors genrated by global_feature_extractor file that you should
import it as library .

Example:
        $ python example_google.py


Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * import features
    * giving single image from user 
    * comparing the single image vector with all of feature vectors
    * computing the MSE and SSIM 
"""
import pandas as pd
import cv2
import numpy as np
import global_feature_extractor as fe
import pickle as pkl 
from skimage.measure import compare_ssim as ssim

#image_path = '/home/mahdi/Pictures/test/1.jpeg'
#image_path = input('please input your image address :  ')
image_size = tuple((500, 500))

with open('features.pkl','rb') as f :
    features = pkl.load(f)


with open('image_path_list.pkl','rb') as f:
    image_path_list = pkl.load(f)
    
    
    
fe_obj = fe.Global_feature_extraction()
counter = 0
mse_all = []
ssim_all = [] 

#
def compare(single_feature):
    """comparing with single feature and all of feature vectors

    Args:
        param1(type==image_vector): single image features that returned from input_image() function.

    Returns:
        param1(type==list(2D)) : returens result of mean squared error algorithm
        param2(type==list(2D)) : returens result of structural similarity algorithm

    """
    for i in range(0,len(features)):
        feature = features[i]
        im_path = image_path_list[i]
        feature=np.array(feature)
        mse = np.square(np.subtract(single_feature, feature)).mean()
        ssim_compare = ssim(single_feature,feature)
        mse_all.append([im_path,mse])
        ssim_all.append([im_path,ssim_compare])
    return mse_all,ssim_all


#
def input_image(image_path):
    Image = cv2.imread(image_path)
    Image = cv2.resize(Image,image_size)
    shape = fe_obj.shape(Image)
    texture   = fe_obj.texture(Image)
    color  = fe_obj.color(Image)
    global_feature = np.hstack([color, texture, shape])
    return global_feature

#
def run():
    """running all codes and save results on pandas dataframe.

    Returns:
        param1(type==dataframe(2D)) : returens result of mean squared error algorithm
        param2(type==dataframe(2D)) 0: returens result of structural similarity algorithm

    """
    image_path = input('please input your image address :  ')
    single_feature = input_image(image_path)
    mse_all ,ssim_all = compare(single_feature)
    df_mse = pd.DataFrame(mse_all,index=[image_path_list])
    df_mse[1] = df_mse[1].sort_index()
    df_ssim = pd.DataFrame(ssim_all,index=[image_path_list])
    df_ssim[1] = df_ssim[1].sort_index()

    # scale features in the range (0-1)
#    scaler = MinMaxScaler(feature_range=(-1, 1))
#    mse_all_scale = scaler.fit_transform([mse_all])
    return df_mse,df_ssim

    
df_mse,df_ssim = run()


with open('data_mse.pkl','wb') as f:
    pkl.dump(df_mse,f)
    
with open('data_ssim.pkl','wb') as f:
    pkl.dump(df_ssim,f)
print('\n\n[STATUS] :   every thing is OK and image compared .  you best :) ')