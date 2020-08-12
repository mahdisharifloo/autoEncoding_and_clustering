import tensorflow as tf 
import pandas as pd
import cv2  
import os
import numpy as np 
from sklearn.cluster import KMeans
# from tensorflow.keras.applications  import vgg19
from tensorflow.keras.applications import nasnet
from scipy import sparse
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd 
from sklearn.cluster import KMeans
import pickle as pkl
import progressbar
from time import sleep
from shutil import copyfile
import argparse
global nasnet_extractor


# from dotenv import load_dotenv
# from pathlib import Path  # python3 only
# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=env_path)


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


if __name__ == "__main__":
    nasnet_extractor = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    parser = argparse.ArgumentParser()
    # parser.add_argument("csv_file", type=str, help="input csv file that you wants cluster it.")
    parser.add_argument("dir_path", type=str, help="input picture directory that you wants cluster it.")
    # parser.add_argument("root_path", type=str, help="image files root path",default='./data/')
    args = parser.parse_args()
    
    dir_path = args.dir_path
    # root_path = args.root_path
    features = {'img':[],'nasnet':[],'cluster':[]}
    # pics_num = os.listdir(img_dir_path)
    # data = pd.read_csv(csv_file)
    data = os.listdir(dir_path)
    pics_num = len(data)
    print('[STATUS] features extraction process running ...')
    bar = progressbar.ProgressBar(maxval=pics_num, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # for i,(file_path,product_id) in enumerate(zip(data['image_path'],data['product_id'])):
    #     img_path = root_path+ file_path
    #     with open(img_path,'rb') as f:
    #         img_bytes = f.read()
    #     Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)

    #     try:
    #         single_feature_table = feature_table_creator(Image)
    #     except:
    #         continue
    #     features['product_id'].append(product_id)
    #     features['img'].append(file_path)
    #     features['nasnet'].append(single_feature_table['nasnet'])
    #     bar.update(i+1)
    
    for i, file_path in enumerate(data):
        img_path = os.path.join(dir_path+file_path)
        with open(img_path,'rb') as f:
            img_bytes = f.read()
        Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
        try:
            single_feature_table = feature_table_creator(Image)
        except Exception as e:
            raise e
        features['img'].append(file_path)
        # features['nasnet'].append(single_feature_table)
        features['nasnet'].append(single_feature_table['nasnet'])
        bar.update(i+1)
    

    print('[STATUS] feature extraction process done')
    if not os.path.exists('db/'):
        os.mkdir('db/')

    n_classes = 20
    res_arr = np.array(features['nasnet'])
    if len(res_arr.shape)==3:
        res_arr = np.reshape(res_arr,(res_arr.shape[0],res_arr.shape[2]))
    print('[STATUS] start kmeans clustering process.it may take few minutes ... ')
    res_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)
    
    for (name,cluster) in zip(features['img'],res_kmeans.labels_):
        features['cluster'].append(cluster)
    df = pd.DataFrame(features)
    print('[STATUS] done')
    kmeans_object_pkl_name = 'db/kmeans_dataset.pkl'
    with open(kmeans_object_pkl_name,'wb') as f:
        pkl.dump(res_kmeans,f)
    print('[STATUS] clustering object saved on {}'.format(kmeans_object_pkl_name))
    
    pkl_file_name = 'db/dataset.pkl'
    with open(pkl_file_name,'wb') as f:
        pkl.dump(features,f)
    print('[STATUS] features saved on {}'.format(pkl_file_name))
    
    print('[STATUS] DONE')
    
    
    
    
    
    
    
    
    
    
