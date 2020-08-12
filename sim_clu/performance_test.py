import os
import tensorflow as tf 
import cv2  
import numpy as np 
from tensorflow.keras.applications.vgg19 import preprocess_input 
from tensorflow.keras.applications import nasnet
import time
import mahotas
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy import sparse
from shutil import copyfile
global data

def nn_image_preprocessing(image_bytes):
    image_bytes = image_bytes[:,:,:3]
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    image_bytes = np.expand_dims(image_bytes, axis=0)
    return image_bytes


def nasnet_feature_extractor(image_bytes):
    preprocess_image = nn_image_preprocessing(image_bytes)
    processed_imgs = nasnet.preprocess_input(preprocess_image)
    # predicting the image
    inception_resnet_v2_features =nasNet.predict(processed_imgs)
    # making features flatten and reshape
    flattened_features = inception_resnet_v2_features.flatten()
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return flattened_features


def feature_table_creator(image_bytes):
    feature_table = {'img':None,'nasnet':None}
    feature_table['nasnet'] = nasnet_feature_extractor(image_bytes)
    return feature_table

def cosin(table1,table2):
    similarity_table = cosine_similarity(table1, table2)
    return  np.mean(similarity_table)


# def cluster_images_finder(cluster_pred,clusters):
#     clusters = os.listdir(cluster_dir_path)
#     if cluster_pred in clusters:
#         files = os.listdir(cluster_dir_path+cluster_pred)
#     return files

def cluster_sim(cluster_pred,single_nasnet_feature_arr,clusters):
    # checking similarity between single image and images that in clusters.
    cluster_images = clusters.loc[clusters['cluster_label'] == cluster_pred]
    compare = {'img':[],'product_id':[],'clu_percent':[]}
    for name,p_id,feature in zip(cluster_images['image'],cluster_images['product_id'],cluster_images['features']):
        percent = cosin(single_nasnet_feature_arr,feature )
        compare['product_id'].append(p_id)
        compare['img'].append(name)
        compare['clu_percent'].append(percent)
    df = pd.DataFrame(compare,index=compare['img'])
    df = df.drop(columns=['img'])
    df = df.sort_values(by='clu_percent',ascending=False, na_position='first')
    return df

if __name__ == "__main__":
    nasNet = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    pkl_kmeans_name = 'db/kmeans_dataset.pkl'
    pkl_file_path = 'db/dataset.pkl'
    cluster_pkl_path = 'db/database_1589551249.5617323.pkl'
    with open(cluster_pkl_path ,'rb') as f:
        clusters = pkl.load(f)
    clusters = pd.DataFrame(clusters)
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
        
    # images = ['/root/ai/evoke_ai/digistyle/crawler/data/category-baby-accessories/data/105759836.jpg',
    #           '/root/ai/evoke_ai/digistyle/crawler/data/category-baby-accessories/data/105777379.jpg',
    #           '/root/ai/evoke_ai/digistyle/crawler/data/category-baby-accessories/data/105806959.jpg',
    #           '/root/ai/evoke_ai/digistyle/crawler/data/category-baby-accessories/data/105830886.jpg',
    #           '/root/ai/evoke_ai/digistyle/crawler/data/category-baby-accessories/data/105834085.jpg']
    images = ['/home/mahdi/Pictures/b24fb664-3483-4106-a69b-017196750171.jpeg',
              '/home/mahdi/Pictures/download.jpeg',
              '/home/mahdi/Pictures/images.jpeg',
              '/home/mahdi/Pictures/t.jpeg',
              '/home/mahdi/Pictures/t2.jpeg']
    # root_dir = '/root/ai/evoke_ai/digistyle/crawler/data/category-boys-accessories/data/'
    # images = os.listdir(root_dir)
    # print(str(len(images)))
    print('befour loop : ',time.ctime())
    for image_path in images:
        # image_path = root_dir+image_path
        Image = cv2.imread(image_path)
        nasnet_feature_arr = nasnet_feature_extractor(Image)
        cluster_pred = kmeans.predict(nasnet_feature_arr)[0]
        cluster_results = cluster_sim(cluster_pred,nasnet_feature_arr,clusters)
        cluster_results.to_csv('prediction_results.csv')
    print('after loop : ',time.ctime())


