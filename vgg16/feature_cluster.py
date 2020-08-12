import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
import pickle
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans

# global vgg_extractor


pkl_name = input('input pkl file path : ')
with open(pkl_name ,'rb') as f:
    features = pickle.load(f)


n_classes = int(input('input number of clusters : '))

res_arr = np.array(features['vgg16'])
print('start clustering ... ')
res_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)

for (name,cluster) in zip(features['img'],res_kmeans.labels_):
    features['cluster'].append(cluster)
    # print(name,cluster)
    
kmeans_object_pkl_name = 'kmeans_'+pkl_name.split('/')[-1]

with open(kmeans_object_pkl_name,'wb') as f:
    pickle.dump(res_kmeans,f)

csv_name = kmeans_object_pkl_name.split('.')[0]+'.csv'
df = pd.DataFrame(features)
df.to_csv(csv_name)