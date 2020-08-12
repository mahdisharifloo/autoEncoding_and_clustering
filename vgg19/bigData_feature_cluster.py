import pandas as pd
import cv2  
import os
import numpy as np 
import pickle
# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
global features
global data_centroids


def make_cluster(n_clasters,res_arr):

    print('start clustering ... ')
    res_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)
    
    # for (name,cluster) in zip(features['img'],res_kmeans.labels_):
    #     features['cluster'].append(cluster)
    for centroid in res_kmeans.cluster_centers_:
        data_centroids.append(centroid)
        # print(name,cluster)


if __name__=="__main__":  
    pkl_name = input('input pkl file path : ')
    with open(pkl_name ,'rb') as f:
        features = pickle.load(f)
    data_centroids = {'centroids':[],'chunk_num':[]}
    data = features['mobilenet']
    n_classes = int(input('input number of clusters : '))
    n_chunks = int(input('input number of chunks : '))
    chunks_len = int(len(data)/n_chunks)
    print('your data has ',len(data),' images')
    print('you have ',chunks_len,' images in even chunk.')
    # res_arr = np.array(features['mobilenet'])
    chunks = [data[x:x+chunks_len] for x in range(0, len(data), chunks_len)]
    for i,chunk in enumerate(chunks):
        print('chunk number : ',i)
        res_arr = np.array(chunk)
        # make_cluster(n_classes,res_arr)
        chunk_kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(res_arr)
        
        for centroid in chunk_kmeans.cluster_centers_:
            data_centroids['chunk_num'].append(i)
            data_centroids['centroids'].append(centroid)
            
    cent_kmeans = KMeans(n_clusters=int(n_classes/2), random_state=0).fit(data_centroids['centroids'])
        
        # kmeans_object_pkl_name = 'kmeans_chunk_'+str(i)+pkl_name.split('/')[-1]
        # csv_name = kmeans_object_pkl_name.split('.')[0]+'.csv'
        # df = pd.DataFrame(features)
        # df.to_csv(csv_name)
    
    


# kmeans_object_pkl_name = 'kmeans_'+pkl_name.split('/')[-1]

# with open(kmeans_object_pkl_name,'wb') as f:
#     pickle.dump(res_kmeans,f)

