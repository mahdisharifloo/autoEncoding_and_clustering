# -*- coding: utf-8 -*-

from keras.models import load_model 
import h5py

model = load_model('features.h5','w')
model = h5py.File('features.h5')