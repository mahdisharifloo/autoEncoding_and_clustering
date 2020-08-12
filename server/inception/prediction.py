import os
import cv2
import numpy as np 
import pickle as pkl
from tensorflow.keras.applications import inception_v3
import pandas as pd
import time
from shutil import copyfile
import progressbar
from scipy import sparse

# make global variable for optimizing Ram space.
global inception_v3

def nn_image_preprocessing(image_bytes):
    image_bytes = image_bytes[:,:,:3]
    image_size =tuple((224, 224))
    image_bytes = cv2.resize(image_bytes,image_size)
    image_bytes = np.expand_dims(image_bytes, axis=0)
    return image_bytes

def inception_v3_feature_extractor(preprocess_image):
    # preprocessing for image input the vgg_server
    # preprocess_image = nn_image_preprocessing(image_bytes)
    processed_imgs = inception_v3.preprocess_input(preprocess_image)
    # predicting the image
    inception_resnet_v2_features =inception_v3_extractor.predict(processed_imgs)
    # making features flatten and reshape
    flattened_features = inception_resnet_v2_features.flatten()
    flattened_features = np.array(flattened_features)
    flattened_features = flattened_features.reshape(1, -1)
    return sparse.csr_matrix(flattened_features)

def pred_decoder(cluster_pred):
    if cluster_pred[0]==1:
        return 'watch'
    else:
        return 'glasses'

if __name__ == "__main__":
    # import nasnet model for extracrating image features.
    inception_v3_extractor = inception_v3.InceptionV3(weights='imagenet', include_top=False,input_shape=(224, 224, 3))    # import kmeans model for predict clusters
    pkl_kmeans_name = 'kmeans_watch_glasses.pkl'
    with open(pkl_kmeans_name ,'rb') as f:
        kmeans = pkl.load(f)
    # save into this path
    image_path = input('input image address : ')
    # image = cv2.imread(image_path)
    with open(image_path,'rb') as f:
        img_bytes = f.read()
    Image = cv2.imdecode(np.fromstring(img_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
    preprocess_image = nn_image_preprocessing(Image)
    inception_v3_feature_arr = inception_v3_feature_extractor(preprocess_image)
    cluster_pred = kmeans.predict(inception_v3_feature_arr)
    label = pred_decoder(cluster_pred)
    print(label)
    

    