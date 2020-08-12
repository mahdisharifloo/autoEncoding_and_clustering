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

global nasnet_extractor


def NASNetMobile(image_bytes):

    image_batch = np.expand_dims(image_bytes, axis=0)
    processed_imgs = nasnet.preprocess_input(image_batch)
    nasnet_features =nasnet_extractor.predict(processed_imgs)
    flattened_features = nasnet_features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return flattened_features


def feature_table_creator(image_bytes):
    image_size =tuple((224, 224))
    image_bytes = image_bytes[:,:,:3]
    image_bytes = cv2.resize(image_bytes,image_size)
    feature_table = {'nasnet':None}
    # feature_table['vgg'] = vgg(image_bytes)
    feature_table['nasnet'] = NASNetMobile(image_bytes)
    return feature_table


def chunk_extractor(chunk,data_dir_path,chunk_number):
    bar = progressbar.ProgressBar(maxval=len(chunk), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    feature_chunks = {'img':[],'nasnet':[],'cluster':[],'chunk':[]}
    for i,img_path in enumerate(chunk):
        img_path = data_dir_path +'/'+ img_path
        # with open(img_path,'rb') as f:
        #     img_bytes = f.read()
        # Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
        Image = cv2.imread(img_path)
        single_feature_table = feature_table_creator(Image)
        feature_chunks['img'].append(img_path)
        feature_chunks['chunk'].append(chunk_number)
        # features['vgg'].append(single_feature_table['vgg'])
        feature_chunks['nasnet'].append(single_feature_table['nasnet'])  
        bar.update(i+1)
        sleep(0.1)
        
    return feature_chunks

def chunk_cluster(chunk):
    print('start clustering ... ')
    res_arr = np.array(feature_chunks['nasnet'])
    res_arr = np.reshape(res_arr,(np.shape(res_arr)[0],np.shape(res_arr)[2]))
    res_kmeans = kmeans.fit(res_arr)
    for (name,cluster) in zip(feature_chunks['img'],res_kmeans.labels_):
        feature_chunks['cluster'].append(cluster)
    df = pd.DataFrame(feature_chunks)
    df = df.drop(columns='nasnet')
    cent = res_kmeans.cluster_centers_
    return df,cent

def centroid_clustering(centroids):
    kmeans = KMeans(n_clusters=n_classes, random_state=0)
    centroid_kmeans = kmeans.fit(centroids)
    cent_of_cents = centroid_kmeans.cluster_centers_
    return centroid_kmeans

def make_dirs(data,chunk_number):
    
    cluster_dir = 'results/chunk_'+str(chunk_number)+'/'
    os.makedirs(os.path.join((cluster_dir)),exist_ok=True)
    # path_cluster = os.path.join(cluster_dir)
    # path_vgg = os.path.join(vgg_dir)
    for i,row in data.iterrows():
        image_name = 'pred_'+row[0].split('/')[-1]
        cluster_label_path = os.path.join(cluster_dir+str(row[2]))
        if not os.path.exists(cluster_label_path):
            os.mkdir(cluster_label_path)
        dst_cluster = os.path.join(cluster_label_path+'/'+image_name)
        copyfile(row[0], dst_cluster)
        


def merge_clusters(features,centroids):
    chunk_cents = [centroids[x:x+n_classes] for x in range(0, len(centroids), n_classes)]
    cents_labels = [level2_clusters.predict([x]) for x in centroids]
    for (features,chunk_cents) in zip(df,cents):
        l2_cluster_pred = level2_clusters.predict([cents])


if __name__ == "__main__":
    # vgg_model = tf.keras.applications.VGG16(weights='imagenet')
    # vgg_extractor = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
    nasnet_extractor = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    data_dir_path = input('input data dir path : ')
    n_classes = 2 
    chunk_size = 50
    kmeans = KMeans(n_clusters=n_classes, random_state=0)
    data_name_list = os.listdir(data_dir_path) 
    len_data = len(data_name_list)
    # make chunks with 500 data 
    chunks = [data_name_list[x:x+chunk_size] for x in range(0, len_data, chunk_size)]
    chunks_info = {'chunk_number':[],'image_path':[],'cluster_num':[]}
    print('{} chunks found .'.format(len(chunks)))
    chunk_results = []
    for i,chunk in enumerate(chunks):

        print('start feature extracting of chunk number {}'.format(str(i)))
        feature_chunks = chunk_extractor(chunk,data_dir_path,i)
        df,cent = chunk_cluster(feature_chunks)
        chunk_results.append(df)
        if i==0:
            centroids = cent
            chunk_results = df
        else:
            centroids = np.concatenate([centroids,cent])
            chunk_results = pd.concat([chunk_results,df],ignore_index=True)
    
    level2_clusters = centroid_clustering(centroids)
    merge_clusters(features,centroids)
    
    
    
        # make_dirs(df,i)
        
        # pkl_chunk_name = 'chunk_{}'.format(i)+'.pkl'
        # with open(pkl_chunk_name,'wb') as f:
        #     pickle.dump(feature_chunks,f)
                    
    
    # for (cent,cluster) in zip(centroids,centroid_kmeans.labels_):
    #     l.append((cent,cluster))
    
    # l2_cluster_pred = level2_clusters.predict([cent])
    
        