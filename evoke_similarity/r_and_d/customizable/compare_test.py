from compareV2 import Compare
from many_extractorV2 import FeatureExtractor as fe
import pickle as pkl
import pandas as pd 

image_path = '/home/mahdi/Pictures/test/2.jpeg'
fe_obj = fe()
product_id = '1476098'
# feature_table = fe_obj.feature_table(image_path,product_id,'shape','texture','sift')
feature_table = fe_obj.feature_table(image_path,product_id,'shape','texture','color','SIFT','SURF','KAZE')


obj = Compare()
ssim_all = obj.compare(feature_table,'shape','texture','color')


features = ['shape','texture','color']

data_file = 'data.pkl'
with open(data_file,'rb') as f :
    dataList = pkl.load(f)
ssim_all = {'shape':[] ,'texture':[] , 'color':[] }
data = pd.DataFrame(dataList)



for feature in features:
    vector1 = feature_table[feature]
    #print('\n',feature,vector1)
    print(feature,len(vector1))
    for vector2, p_id in zip(data[feature], data['product_id']):
        #print('\n',feature,vector2)
        # print(feature,len(vector2))
        # print(p_id)
        ssim_res = obj.ssim_compare(vector1,vector2)
        ssim_all[feature].append((p_id,ssim_res))