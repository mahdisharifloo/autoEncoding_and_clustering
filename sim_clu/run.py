from functools import wraps
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
from dotenv import load_dotenv
import argparse

load_dotenv(verbose=True)

global data
# tf.config.gpu_options.allow_growth = True

def estaminetor_decorator(func):
    # Without the use of this decorator factory, the name of the example function would have been 'wrapper',
    # https://docs.python.org/3/library/functools.html#functools.wraps
    wraps(func)
    def wrapper(*args, **kwargs):
        # perf_counter has more accuracy
        # https://docs.python.org/3/library/time.html#time.perf_counter
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print("function %s takes %s" % (func.__name__, end-start))
        return result
    return wrapper

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
    compare = {'img':[],'clu_percent':[]}
    for name,feature in zip(cluster_images['image'],cluster_images['features']):
        percent = cosin(single_nasnet_feature_arr,feature )
        compare['img'].append(name)
        compare['clu_percent'].append(percent)
    
    # for i,cluster_number in enumerate(clusters['cluster_label']):
    #     if cluster_number==cluster_pred:
    #             percent = cosin(single_nasnet_feature_arr, clusters['features'][i])
    #             compare['product_id'].append(clusters['product_id'][i])
    #             compare['img'].append(clusters['image'][i])
    #             compare['clu_percent'].append(percent)
    #     else :
    #         continue

    df = pd.DataFrame(compare,index=compare['img'])
    df = df.drop(columns=['img'])
    df = df.sort_values(by='clu_percent',ascending=False, na_position='first')
    return df

nasNet = nasnet.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))

parser = argparse.ArgumentParser()
parser.add_argument("cluster_pkl_path", type=str, help="input pkl file that you wants cluster it.")
parser.add_argument("image_path", type=str, help="image file path")
parser.add_argument("pkl_kmeans_name", help="kmeans object that you train it before", default='./db/kmeans_dataset.pkl')
args = parser.parse_args()


@estaminetor_decorator
def loader():
    pkl_kmeans_name = './db/kmeans_dataset.pkl'
    cluster_pkl_path = os.getenv("CSV_PATH")
    if cluster_pkl_path == None:
        cluster_pkl_path = args.cluster_pkl_path
    with open(cluster_pkl_path ,'rb') as f:
        clusters = pkl.load(f)
    clusters = pd.DataFrame(clusters)
    with open(args.pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)

    return clusters, kmeans

@estaminetor_decorator
def main():

    clusters, kmeans = loader()
    image_path = os.getenv("IMAGE_PATH")
    if image_path == None:
        image_path = args.image_path
    Image = cv2.imread(image_path)
    nasnet_feature_arr = nasnet_feature_extractor(Image)
    cluster_pred = kmeans.predict(nasnet_feature_arr)[0]
    cluster_results = cluster_sim(cluster_pred,nasnet_feature_arr,clusters)
    print(cluster_results.head(number_of_item))
    cluster_results.to_csv('prediction_results.csv')

main()

