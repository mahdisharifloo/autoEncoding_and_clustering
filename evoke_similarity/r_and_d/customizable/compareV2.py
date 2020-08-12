# -*- coding: utf-8 -*-
"""
this module compairing two image feature vectors.

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
import cv2


class Compare:
    def __init__(self,data_file='data.pkl'):
        self.image_size = tuple((500, 500))
        with open(data_file,'rb') as f :
            self.dataList = pkl.load(f)
        self.similarityALL = {'shape':[] ,'texture':[] , 'color':[],'SIFT':[],'SURF':[],'KAZE':[],'ORB':[]}
        self.data = pd.DataFrame(self.dataList)

    def mse_compare(self,vector1,vector2):
        mse = np.square(np.subtract(vector1, vector2)).mean()
        return mse
    
    def ssim_compare(self,vector1,vector2):
        ssim_compare = ssim(vector1,vector2)
        return ssim_compare
    
    def cosin(self,table1,table2):
        similarity_table = cosine_similarity(table1, table2)
        similarity_percent = np.mean(similarity_table)
        return similarity_percent

    # compare(ssim,shape,sift,orb)
    def compare(self,single_feature_table,*features):
        for feature in features:
            vector1 = single_feature_table[feature]
            # print(vector1)
            for vector2,p_id  in zip(self.data[feature],self.data['product_id']):
                # print(vector2)
                if feature == 'SIFT' or feature =='SURF'or feature =='KAZE' or feature =='ORB':
                    print('comparing with cosin similarity')
                    cosin_res = self.cosin(vector1, vector2)
                    self.similarityALL[feature].append((p_id,cosin_res))

                else :
                    print('comparing with ssim')
                    ssim_res = self.ssim_compare(vector1,vector2)
                    self.similarityALL[feature].append((p_id,ssim_res))

        return self.similarityALL
            
    # def macher(self,des1,des2):
    #     # good = []
    #     FLANN_INDEX_KDTREE = 1
    #     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #     search_params = dict(checks = 50)
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     matches = flann.knnMatch(des1,des2,k=2)
    #     # for m,n in matches:
    #     #     if m.distance < 0.7*n.distance:
    #     #         good.append(m)
    #     self.matched.append(len(matches))
    #     return self.matched