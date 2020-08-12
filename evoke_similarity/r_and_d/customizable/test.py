

# libraris 
import numpy as np
import cv2
import features as fe
import glob
import pickle as pkl
import pandas as pd
import json

image_path2 = '/home/mahdi/Pictures/test/2.jpeg'
image_path = '/home/mahdi/Pictures/test/3.jpeg'

image_size = tuple((500,500))
with open('conf/feature_names.json') as f:    
    features_name = json.load(f)  

fe_glb = fe.Global_feature_extraction()

args = ['shape','texture','sift']
feature_table = {'shape':None,'texture':None,'color':None,'SIFT':None,'SURF':None,'KAZE':None,'Dense':None}
Image = cv2.imread(image_path)
Image = cv2.resize(Image,image_size)
Shape = getattr(fe_glb,features_name['shape'])

for ar in args:
    if ar == 'shape':
        print('ok')
        shape = Shape(Image)
        feature_table['shape']=shape   



        # self.data = pd.read_csv(self.csv_path,index_col=0)
        # self.Shape = getattr(fe_glb,features_name['shape'])
        # self.Texture = getattr(self.fe_glb,self.features_name['texture']) 
        # self.Color = getattr(self.fe_glb,self.features_name['color'])
        # self.Sift = getattr(self.fe_lcl,self.features_name['SIFT'])
        # self.Surf = getattr(self.fe_lcl,self.features_name['SURF'])
        # self.Orb = getattr(self.fe_lcl,self.features_name['ORB'])
        # self.brief = getattr(self.fe_lcl,self.features_name['SIFT'])
        # self.Kaze = getattr(self.fe_lcl,self.features_name['KAZE'])
        

from many_extractorV2 import FeatureExtractor as fe

fe_obj = fe()
feature_table = fe_obj.feature_table(image_path,'shape','texture','sift')
feature_table2 = fe_obj.feature_table(image_path2,'shape','texture','sift')

from sklearn.metrics.pairwise import cosine_similarity 
cosSimilarities = cosine_similarity(feature_table['texture'].flatten())




with open('conf/conf.json') as f:    
    config = json.load(f)
csv_path  = config["csv_path"]
data = pd.read_csv(csv_path,index_col=0)

features = fe_obj.csv_extractor()

fe_obj.save()


features = []
for img_path in data['image_path']:
    print('ok')
    feature_table = fe_obj.feature_table(img_path,'shape','texture','sift')
    features.append(feature_table)

data['feature_vector']= features



