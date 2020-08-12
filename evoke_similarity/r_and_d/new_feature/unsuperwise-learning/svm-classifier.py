# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl 
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
clf = OneVsOneClassifier(LinearSVC(random_state=0))
labels_words = res.labels_
def _encodeLabels(labels_words):
    le.fit(labels_words)
    return np.array(le.transform(labels_words), dtype=np.float32)
y = _encodeLabels(labels_words)

for i in range(0,len(centroids)):
    centroid = centroids[i]
    im_path = image_path_list[i]
    feature=np.array(feature)

svm = ClassifierTrainer(centroids, labels_words)
X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]