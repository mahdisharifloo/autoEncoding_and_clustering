# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np 
import pickle as pkl
from tensorflow.keras.applications import nasnet
import pandas as pd
import time
from shutil import copyfile
import progressbar
from scipy import sparse

# make global variable for optimizing Ram space.
global nasNet

def nn_image_preprocessing(image_bytes):
    image_bytes = image_bytes[:,:,:3]
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    image_bytes = np.expand_dims(image_bytes, axis=0)
    return image_bytes

def nasnet_feature_extractor(preprocess_image):
    # preprocessing for image input the vgg_server
    # preprocess_image = nn_image_preprocessing(image_bytes)
    processed_imgs = nasnet.preprocess_input(preprocess_image)
    # predicting the image
    inception_resnet_v2_features =nasNet.predict(processed_imgs)
    # making features flatten and reshape
    flattened_features = inception_resnet_v2_features.flatten()
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return sparse.csr_matrix(flattened_features)




def make_dirs(data,unique_name):
    
    cluster_dir = 'results/'+unique_name+'/'
    os.makedirs(os.path.join((cluster_dir)),exist_ok=True)
    # path_cluster = os.path.join(cluster_dir)
    # path_vgg = os.path.join(vgg_dir)
    for i,row in data.iterrows():
        image_name = 'pred_'+row[0].split('/')[-1]
        cluster_label_path = os.path.join(cluster_dir+str(row[1]))
        if not os.path.exists(cluster_label_path):
            os.mkdir(cluster_label_path)
        dst_cluster = os.path.join(cluster_label_path+'/'+image_name)
        copyfile(row[0], dst_cluster)
        

if __name__ == "__main__":
    # import nasnet model for extracrating image features.
    nasNet = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # import kmeans model for predict clusters
    pkl_kmeans_name = input('input kmeans object pkl file : ')
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
    # save into this path

    # giving data directory that want to classify its data
    #/home/mahdi/projects/dgkala/cluster_inception_resnet_v2/data/ring_glasses_hand_watch/
    data_dir = input('input directory address : ')
    if data_dir[-1]!='/':
        data_dir = data_dir+'/'
    data_names = os.listdir(data_dir)
    len_data = len(data_names)
    print('[STATUS] ',len_data,'data founded .\n')
    # dictionary for prediction
    prediction = {'image':[],'cluster_label':[]}
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,file_name in enumerate(data_names):
        file = data_dir+file_name
        image = cv2.imread(file)
        # if image was .PNG this works
        try:
            preprocess_image = nn_image_preprocessing(image)
        except:
            continue
        # make prediction with clustring model.
        nasnet_feature_arr = nasnet_feature_extractor(preprocess_image)
        cluster_pred = kmeans.predict(nasnet_feature_arr)
        # cluster_label = cluster_pred_decoder(cluster_pred)
        # add result to dictionary 
        prediction['image'].append(file)
        prediction['cluster_label'].append(cluster_pred[0])

        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    unique_time = str(time.time())
    csv_name = unique_time+'.csv'
    df.to_csv(csv_name)
    make_dirs(df,unique_time)
