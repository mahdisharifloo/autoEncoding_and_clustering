

import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications import vgg19
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans
import pickle as pkl
import progressbar
from time import sleep
# global vgg_extractor
global vgg19_extractor


def Vgg19(image_bytes):

    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = vgg19.preprocess_input(image_batch)
    vgg19_features =vgg19_extractor.predict(processed_imgs)
    flattened_features = vgg19_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    return flattened_features


def feature_table_creator(image_bytes):
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    feature_table = {'vgg19':None}
    # feature_table['vgg'] = vgg(image_bytes)
    feature_table['vgg19'] = Vgg19(image_bytes)
    return feature_table


if __name__ == "__main__":
    # vgg_model = tf.keras.applications.VGG16(weights='imagenet')
    # vgg_extractor = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    vgg19_extractor = vgg19.VGG19(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

    pkl_kmeans_name = input('input kmaens opject pkl file path : ')
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
    
    
    
    image = cv2.imread(input('input single image for predicting cluster : '))
    image = image[:,:,:3]
    single_feature_table = feature_table_creator(image)
    single_res_arr = np.array(single_feature_table['vgg19'])
    single_res_arr = single_res_arr.reshape(1, -1)
    pred = kmeans.predict(single_res_arr)
    print(pred)

    