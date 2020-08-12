# -*- coding: utf-8 -*-
import pandas as pd 
import os 
from shutil import copyfile
csv_file_name = input('input inception_resnet_v2 csv kmeans : ')
data = pd.read_csv(csv_file_name)
data = data.drop(['inception_resnet_v2','Unnamed: 0'],axis=1)
data = data.sort_values(by=['cluster'])


for row in data.iterrows():
    path = os.path.join("clusters/{}".format(row[1][1]))
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
        print(path,'created')
        
    if os.path.exists(path):
        print(path + ' : exists')
    dst = path+'/'+row[1][0].split('/')[-1]
    copyfile(row[1][0], dst)
