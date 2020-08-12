#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:49:07 2019

@author: amir
"""
import pickle
import numpy as np
import pandas as pd

def cont(records, item):
    image_path = 'image path'
    #print('item[1][compare_val]', type(item), item)

    #print("item is", type(item), item)
    for key, record in records.iterrows():
        # print("record", type(record[0][0]), record[0][0])
        #print('record[1][compare_val]', type(record[1][compare_val]), record[1][compare_val])
        if (record[image_path] == item[image_path]) :
            # print("item is", type(item), item)
            # print("record", type(record[0][0]), record[0][0])

            return True
#        print(record[image_path], item[image_path])
    
    return False

        
        
    
def add(l, item):
    
    pass
# def take(l)
    
def main():
    
    with open('data_mse.pkl','rb') as f :
        mse = pickle.load(f)
    
    with open('data_ssim.pkl','rb') as f :
        ssim = pickle.load(f)
    
    compare_val = "compare val"
    df_columns = ('Index', 'image path', 'compare val')

    # largest value from second column which is similarity percent
    # largest mean higer similarity
    result_list = ssim.nlargest(10, compare_val)
    
    # smallest diffrece in mse
    smallest_mse = mse.nsmallest(5, compare_val)
    
    
    for key, smol in smallest_mse.iterrows():
        #smol = smallest_mse.iloc[[smol_index]]
        
        in_list = cont(result_list, smol)
        # print(smol, "in the resllt list", in_list)
            
        if(not in_list):
            print("not in list")
            # this goint to be valeu for index, image path, compare val
            data = [smol[df_columns[1]], smol[df_columns[2]]]
            smol_dataframe = pd.DataFrame([data], columns=df_columns[1:])
            result_list = result_list.append(smol_dataframe)
            # result_list.append(smol, verify_integrity=True)
            # print("add item", smol, "to resutl list")
        
    return result_list
    
    
if __name__ == "__main__":
    result_list = main()
