# -*- coding: utf-8 -*-

import pandas as pd
import cv2  
import os
import numpy as np 
import pickle
# from tensorflow.keras.applications  import vgg19
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import MeanShift



pkl_name = input('input pkl file path : ')

with open(pkl_name ,'rb') as f:
    features = pickle.load(f)


# n_classes = int(input('input number of clusters : '))

res_arr = np.array(features['mobilenet'])
print('start clustering ... ')
res_meanshift = MeanShift(bandwidth=500).fit(res_arr)

for (name,cluster) in zip(features['img'],res_meanshift.labels_):
    features['cluster'].append(cluster)
    # print(name,cluster)
df = pd.DataFrame(features)


meanshift_object_pkl_name = 'meanshift_'+pkl_name.split('/')[-1]

with open(kmeans_object_pkl_name,'wb') as f:
    pickle.dump(res_meanshift,f)

csv_name = meanshift_object_pkl_name.split('.')[0]+'.csv'

df.to_csv(csv_name)