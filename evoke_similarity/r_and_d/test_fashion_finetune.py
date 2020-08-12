# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.models import load_model 

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
from tensorflow.python.framework import ops
# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]
image_path1 = input('please input first image : ')
image_path2 = input('pease input second image : ')


base_model = load_model('model7.h5')
model = Model(input=base_model.input, output=base_model.get_layer('res5c_branch2c').output)
image_size = (224, 224)


features = []

def extractor(image_path,image_size):
    image_file = cv2.imread(image_path)
    image_file = cv2.resize(image_file, image_size)
    x = preprocess_input(image_file)
    x = np.expand_dims(x, axis=0)
    feature = model.predict(x)
    flat = feature.flatten()
    return flat


flat1 = extractor(image_path1,image_size)
flat2 = extractor(image_path2,image_size)


mse = np.square(np.subtract(flat1,flat2)).mean()

print(mse)