# -*- coding: utf-8 -*-


import pickle as pkl 
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pkl_path", type=str, help="input pkl file that you wants split it clusters.")
args = parser.parse_args()


with open(args.pkl_path ,'rb') as f:
    data = pkl.load(f)
    
df = pd.DataFrame(data)
df = df.sort_values(by=['cluster_label'])
groups = df.groupby(df['cluster_label'])
group_names = list(groups.groups.keys())
print('this groups founded :  {}'.format(group_names) )
cluster_groups = [groups.get_group(x) for x in groups.groups]

for cluster_split in cluster_groups:
    cluster_split = cluster_split.reset_index()
    pkl_name = 'cluster_'+str(cluster_split['cluster_label'].unique()[0])+'_DB.pkl'
    with open(pkl_name,'wb') as f:
        pkl.dump(cluster_split,f)


