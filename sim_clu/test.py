# -*- coding: utf-8 -*-

import pickle as pkl 
import pandas as pd 


first_pkl_path = input('input first pkl database : ')
with open(first_pkl_path ,'rb') as f:
    data = pkl.load(f)