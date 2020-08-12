# -*- coding: utf-8 -*-

import numpy as np
import cv2
import global_feature_extractor as fe
import os
import glob
import pickle as pkl
import pandas as pd
import global_feature_extractor as fe
from many_extractor import FeatureExtractor


ex = FeatureExtractor()

data = ex.csv_extractor()
image_path_list = list(data['image_path'])
features = list(data['feature_vector'])
product_id = list(data['product_id'])


csv_path='image_data.csv' 
data = pd.read_csv(csv_path,index_col=0)
data = data.head()
def csv_extractor():
    features = []
    for img_path in data['image_path']:
        global_feature = ex.img2vector(img_path)
        features.append(global_feature)
    data['feature_vector']= features
    return data


with open('mse_all.pkl','rb') as f:
   mse_all = pkl.load(f)