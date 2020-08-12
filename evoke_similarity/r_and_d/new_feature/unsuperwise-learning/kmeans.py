# -*- coding: utf-8 -*-

''' 
this madule make kmeans clustering for unsupervise learning on our feature data

'''
import pickle as pkl
import cv2 
import numpy as np 
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from many_extractor import FeatureExtractor as fext


single_image_path = '/home/mahdi/Pictures/test/8.png'
#create object of feature extractor on many_extractor file to make vector of single image.
fextOBJ = fext()
single_feature = fextOBJ.img2vector(single_image_path)
feature_file='features.pkl'
image_path_file='image_path_list.pkl'
num_clusters=32
num_retries = 10


def input_dataframes():
    '''
    load .pkl data on dataframe and return it.
    '''
    with open(feature_file,'rb') as f :
        features = pkl.load(f)
    with open(image_path_file,'rb') as f:
        image_path_list = pkl.load(f)
    return features, image_path_list

def pca():
    pass

def kmeans_cluster(features):
    '''
    make cluster of feaature data and make segment of feature images.
    we used KMEANS algorithm to do this for us.
    '''
    # Create KMeans object
    kmeans = KMeans(num_clusters,n_init=max(num_retries, 1),max_iter=10, tol=1.0)
    # Run KMeans on the datapoints
    res = kmeans.fit(features)
    # Extract the centroids of those clusters
    centroids = res.cluster_centers_
    return res,kmeans, centroids

features, image_path_list= input_dataframes()

res,kmeans,centroids = kmeans_cluster(features)

#feature_map = extract_feature_map(input_map, kmeans, centroids)

#with open("kmeas.pkl", 'w') as f:
#    pkl.dump((kmeans, centroids), f)

#
#
#for i in range(0,len(features)):
#    feature = features[i]
#    im_path = image_path_list[i]
#    feature=np.array(feature)