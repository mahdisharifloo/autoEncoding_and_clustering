# -*- coding: utf-8 -*-
import pandas as pd
import cv2  
import os
import numpy as np 
import pickle
import pandas as pd 
from sklearn.cluster import KMeans
from tensorflow.keras.applications import nasnet
import progressbar
from time import sleep
from shutil import copyfile
import time


pkl_name = 'chunk_6.pkl'
with open(pkl_name ,'rb') as f:
    chunk6 = pickle.load(f)

l = [] 

for (cent,cluster) in zip(centroids,centroid_kmeans.labels_):
    l.append((cent,cluster))