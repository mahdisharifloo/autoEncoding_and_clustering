# -*- coding: utf-8 -*-
"""
this module compairing twe image feature vectors.

NOTES :     
    
    - feature extractor algortithms genrates data by global_feature_extractor.
    - feature vector generated by many_extractorV2 that save on .pkl files and you should import them on __init__() function.
    - this class madule can compare two vector by three algortims :
        - mean square error
        - strunctured similarity 
        - cosin similarity 

Todo:
    * import features
    * giving single image from other file that import this madule
    * comparing the single image vector with all of feature vectors
    * computing the MSE and SSIM and cosin
"""
import numpy as np
import pickle as pkl 
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Compare:
    def __init__(self,feature_file='features.pkl',image_path_file='image_path_list.pkl'):
        self.image_size = tuple((500, 500))
        with open(feature_file,'rb') as f :
            self.features = pkl.load(f)
        with open(image_path_file,'rb') as f:
            self.image_path_list = pkl.load(f)
    
        self.counter = 0
        self.mse_all = []
        self.ssim_all = [] 
        
    def mse_compare(self,vecor1,vector2):
        mse = np.square(np.subtract(vecor1, vector2)).mean()
        return mse
    
    def ssim_compare(self,vecor1,vector2):
        ssim_compare = ssim(vecor1,vector2)
        return ssim_compare
    
    def cosin_compare(self,vecor1):
        ### it has ERROR don't use this function yet 
        cosine_similarity(vecor1, self.features)
        return cosine_similarity
    
    def create_dataframe( self,mse_all ,ssim_all ):
        
        df_mse = pd.DataFrame(mse_all,columns=['image path','compare val'])
        df_ssim = pd.DataFrame(ssim_all,columns=['image path','compare val'])
        df_mse = df_mse.sort_values(by=['compare val'])
        df_ssim = df_ssim.sort_values(by=['compare val'],ascending=False, na_position='first')
        df_mse = df_mse.reset_index(drop=True)
        df_ssim = df_ssim.reset_index(drop=True)
        return df_mse,df_ssim
        
        return df_mse,df_ssim
    def save(self ,df_mse,df_ssim,df_cosin,
             file_mse='mse_all.pkl' , 
             file_ssim='ssim_all.pkl',
             file_cosin='cosin_all.pkl'):
        
        with open(file_mse,'wb') as f:
            pkl.dump(df_mse,f)
            
        with open(file_ssim,'wb') as f:
            pkl.dump(df_ssim,f) 
            
        with open(file_cosin,'wb') as f:
            pkl.dump(df_cosin,f) 