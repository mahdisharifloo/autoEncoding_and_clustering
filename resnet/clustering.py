# -*- coding: utf-8 -*-

import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications  import resnet50
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans
# global vgg_extractor
global resnet50_extractor

# def vgg(image_bytes):
#     image_batch = np.expand_dims(image_bytes, axis=0)
#     processed_imgs = vgg19.preprocess_input(image_batch)
#     vgg_features = vgg_extractor.predict(processed_imgs)
#     return np.transpose(vgg_features)

def Resnet50(image_bytes):

    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = resnet50.preprocess_input(image_batch)
    resnet50_features =resnet50_extractor.predict(processed_imgs)
    flattened_features = resnet50_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    return flattened_features


def feature_table_creator(image_bytes):
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    feature_table = {'resnet':None}
    # feature_table['vgg'] = vgg(image_bytes)
    feature_table['resnet'] = Resnet50(image_bytes)
    return feature_table

# def cluster():
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(array)


if __name__ == "__main__":
    # vgg_model = tf.keras.applications.VGG16(weights='imagenet')
    # vgg_extractor = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    resnet50_extractor = resnet50.ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    img_dir_path = input('[INPUT] image dir path : ')
    features = {'img':[],'resnet':[],'cluster':[]}
    pics_num = os.listdir(img_dir_path)
    for i,img_path in enumerate(pics_num):
        img_path = img_dir_path + img_path
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
        single_feature_table = feature_table_creator(Image)
        features['img'].append(img_path)
        # features['vgg'].append(single_feature_table['vgg'])
        features['resnet'].append(single_feature_table['resnet'])
        print(str(int(i/len(pics_num)*100)))
    res_arr = np.array(features['resnet'])
    # gg_arr = np.array(features['vgg'])
    res_kmeans = KMeans(n_clusters=30, random_state=0).fit(res_arr)
    # vgg_kmeans = KMeans(n_clusters=2, random_state=0).fit(vgg_arr)
    # X = single_feature_table['resnet']
    for (name,cluster) in zip(features['img'],res_kmeans.labels_):
        features['cluster'].append(cluster)
        # print(name,cluster)
    df = pd.DataFrame(features)
    df.to_csv('result.csv')
    
    