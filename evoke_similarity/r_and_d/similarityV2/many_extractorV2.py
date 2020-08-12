# libraris 


import numpy as np
import cv2
import global_feature_extractor as fe
import os
import glob
import pickle as pkl



class FeatureExtractor:
    def __str__():
        pass
    def __init__(self,  dir_path='dataset/data' ):
        self.image_size = tuple((500, 500))
        self.features = []
        self.image_path_list = []
        self.mse_all = []
        self.labels = []
        #object of feature extractor
        self.fe_obj = fe.Global_feature_extraction()
        self.dir_path = dir_path
        self.lables = os.listdir(dir_path)
        self.lables.sort()
        
    def img2vector(self,image_path):
        
        Image = cv2.imread(image_path)
        Image = cv2.resize(Image,self.image_size)
        shape = self.fe_obj.shape(Image)
        texture   = self.fe_obj.texture(Image)
        color  = self.fe_obj.color(Image)
        global_feature = np.hstack([color, texture, shape])
        return global_feature
        
    def extractor(self):
        # loop over all the labels in the folder
        count = 1
        features = []
        labels = []
        image_path_list = []
        for i, label in enumerate(self.lables):
            cur_path = self.dir_path + "/" + label
            count = 1
            for image_path in glob.glob(cur_path + "/*.jpg"):
                global_feature = self.img2vector(image_path)
                features.append(global_feature)
                labels.append(label)
                image_path_list.append(image_path)
                print("[INFO] processed - " + str(count))
                count += 1
        print("[INFO] completed label - " + label)
        return features,labels,image_path_list



    def save(self,features,image_path_list,
             feature_file_name='features.pkl',
             imgPathList_fileName='image_path_list.pkl'):
        
        with open(feature_file_name,'wb') as f:
            pkl.dump(features,f)
        with open(imgPathList_fileName,'wb') as f:
            pkl.dump(image_path_list,f)
    