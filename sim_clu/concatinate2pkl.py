# -*- coding: utf-8 -*-

import pickle as pkl 

first_pkl_path = input('input first pkl database : ')
with open(first_pkl_path ,'rb') as f:
    pkl1 = pkl.load(f)
    
second_pkl_path = input('input second pkl database : ')
with open(second_pkl_path ,'rb') as f:
    pkl2 = pkl.load(f)
    

res = {'image':[],'features':[],'product_id':[],'cluster_label':[]}



res['image'] = pkl1['image']+pkl2['image']
res['features'] = pkl1['features']+pkl2['features']
res['product_id'] = pkl1['product_id']+pkl2['product_id']
res['cluster_label'] = pkl1['cluster_label']+pkl2['cluster_label']


pkl_name = input('input destination pkl file : ')
with open(pkl_name,'wb') as f:
    pkl.dump(res,f)
