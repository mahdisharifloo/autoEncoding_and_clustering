# -*- coding: utf-8 -*-
import pickle
import numpy as np 
from scipy import sparse
import pandas as pd





pkl_name = input('input pkl file path : ')
with open(pkl_name ,'rb') as f:
    features = pickle.load(f)

data = np.array(features['nasnet'])




from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
clu = cluster.fit(data)
