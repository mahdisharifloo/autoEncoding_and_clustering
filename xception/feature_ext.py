# -*- coding: utf-8 -*-


import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications import xception
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
global xception_extractor


def Xception(image_bytes):

    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = xception.preprocess_input(image_batch)
    xception_features =xception_extractor.predict(processed_imgs)
    flattened_features = xception_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    return flattened_features


def feature_table_creator(image_bytes):
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    feature_table = {'xception':None}
    # feature_table['vgg'] = vgg(image_bytes)
    feature_table['xception'] = Xception(image_bytes)
    return feature_table


if __name__ == "__main__":
    # vgg_model = tf.keras.applications.VGG16(weights='imagenet')
    # vgg_extractor = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    xception_extractor = xception.Xception(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    img_dir_path = input('[INPUT] image dir path : ')
    features = {'img':[],'xception':[],'cluster':[]}
    pics_num = os.listdir(img_dir_path)
    bar = progressbar.ProgressBar(maxval=len(pics_num), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i,img_path in enumerate(pics_num):
        img_path = img_dir_path + img_path
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
        Image = Image[:,:,:3]
        single_feature_table = feature_table_creator(Image)
        features['img'].append(img_path)
        # features['vgg'].append(single_feature_table['vgg'])
        features['xception'].append(single_feature_table['xception'])
        bar.update(i+1)
        sleep(0.1)
    pkl_file_name = img_dir_path.split('/')[-2]+'.pkl'
    with open(pkl_file_name,'wb') as f:
        pkl.dump(features,f)
        
        