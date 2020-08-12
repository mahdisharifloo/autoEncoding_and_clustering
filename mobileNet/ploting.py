
import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications import mobilenet
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans
import pickle as pkl
import progressbar
from time import sleep
%matplotlib inline


pkl_features_name = input('input pkl feature names : ')
pkl_kmeans_name = input('input kmaens opject pkl file path : ')
with open(pkl_kmeans_name ,'rb') as f:
    kmeans = pkl.load(f)
with open(pkl_features_name ,'rb') as f:
    features = pkl.load(f)



from sklearn.decomposition import PCA
pca = PCA(.95)
pca_result = pca.fit(features['mobilenet'])
    



from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],
            c='black', s=200, alpha=0.5);



plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
# plt.subplot(211)
plt.plot(centers[0],'g^',centers[1],'bo')
plt.savefig('test.png',dpi=100)


#set font size of labels on matplotlib plots
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

n = kmeans.n_clusters

groups = {'cluster_1': kmeans.cluster_centers_[0],
          'cluster_2': kmeans.cluster_centers_[1]}
data = pd.DataFrame(index=range(n*len(groups)), columns=['x','y','label'])

for i, group in enumerate(groups.keys()):
    #randomly select n datapoints from a gaussian distrbution
    data.loc[i*n:((i+1)*n)-1,['x','y']] = np.random.normal(groups[group], 
                                                           [0.5,0.5], 
                                                           [n,2])