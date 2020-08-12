# -*- coding: utf-8 -*-
import pickle as pkl
from many_extractorV2 import FeatureExtractor as fe
fe_obj = fe()



def save(destination_file_name):
    features = fe_obj.csv_extractor()
    with open(destination_file_name,'wb') as f:
        pkl.dump(features,f)

destination_file_name = input('input your destination pickle file name : ')
save()