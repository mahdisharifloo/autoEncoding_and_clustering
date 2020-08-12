# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import cv2
import numpy as np 
import pickle as pkl
from tensorflow.keras.applications import nasnet
import pandas as pd
import time
import progressbar
from scipy import sparse
import argparse

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




if __name__ == "__main__":
    # import nasnet model for extracrating image features.
    nasNet = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # import kmeans model for predict clusters
    pkl_kmeans_name = 'db/kmeans_dataset.pkl'
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
    # save into this path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file_path", type=str, help="input csv file that you wants cluster it.")
    parser.add_argument("root_dir", type=str, help="input root directory.",default='.')    
    args = parser.parse_args()
    
    # giving data directory that want to classify its data
    csv_file_path = args.csv_file_path
    data = pd.read_csv(csv_file_path)
    root_dir = args.root_dir
    len_data = len(data)
    print('[STATUS] ',len_data,'data founded .\n')
    # dictionary for prediction
    prediction = {'image':[],'product_id':[],'cluster_label':[],'features':[]}
    # show progressbar for make process visual 
    bar = progressbar.ProgressBar(maxval=len_data, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for loop on data
    for i,(file_name,p_id) in enumerate(zip(data['image_path'],data['product_id'])):
        file_ = root_dir +file_name
        image = cv2.imread(file_)
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
        prediction['features'].append(nasnet_feature_arr)
        prediction['image'].append(file_name)
        prediction['cluster_label'].append(cluster_pred[0])
        prediction['product_id'].append(p_id)
        bar.update(i+1)
        
    print('process done.\nsaving results ...\n every things DONE .')
    # make dataframe and csv file and save the predictions
    df = pd.DataFrame(prediction)
    groups = df.groupby(df['cluster_label'])
    group_names = list(groups.groups.keys())
    print('this groups founded :  {}'.format(group_names) )
    cluster_groups = [groups.get_group(x) for x in groups.groups]
    
    for cluster_split in cluster_groups:
        cluster_split = cluster_split.reset_index()
        pkl_name = 'db/'+'cluster_'+str(cluster_split['cluster_label'].unique()[0])+'_DB.pkl'
        with open(pkl_name,'wb') as f:
            pkl.dump(cluster_split,f)
    

    
    
    # df = df.sort_values(by='cluster_label')
    # labels =  df['cluster_label'].unique().tolist()
    # unique_time = str(time.time())
    # pkl_name = 'db/database_'+unique_time+'.pkl'
    # with open(pkl_name,'wb') as f:
    #     pkl.dump(prediction,f)
