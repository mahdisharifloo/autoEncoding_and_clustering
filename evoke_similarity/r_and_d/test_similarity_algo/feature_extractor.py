#|_________________________|
#|wirten by mahdi sharifloo|
#|_________________________|

import numpy as np
import cv2
import global_feature_extractor as fe
import os
import glob
import pickle as pkl
#******************************  hiper parameters
image_size = tuple((500, 500))
features = []
image_path_list = []
mse_all = []
labels = []
#object of feature extractor
fe_obj = fe.Global_feature_extraction()
dir_path = 'dataset/data'
lables = os.listdir(dir_path)
lables.sort()

#****************************** 
print("\n\n [INFO] successfully loaded hiper parameters ...")


# loop over all the labels in the folder
count = 1
for i, label in enumerate(lables):
  cur_path = dir_path + "/" + label
  count = 1
  features = []
  labels = []
  image_path_list = []
  for image_path in glob.glob(cur_path + "/*.jpg"):
    Image = cv2.imread(image_path)
    Image = cv2.resize(Image,image_size)
    shape = fe_obj.shape(Image)
    texture   = fe_obj.texture(Image)
    color  = fe_obj.color(Image)
    global_feature = np.hstack([color, texture, shape])
#    arr_feature['name'] ,arr_feature['feature_vector'] = image_path,global_feature
    #features.append((image_path,global_feature))
    features.append(global_feature)
    labels.append(label)
    image_path_list.append(image_path)
    
    print("[INFO] processed - " + str(count))
    count += 1
  print("[INFO] completed label - " + label)


with open('features.pkl','wb') as f:
    pkl.dump(features,f)


with open('image_path_list.pkl','wb') as f:
    pkl.dump(image_path_list,f)
    
    
print('\n\n[STATUS] :   every thing is OK and features extracted .  you best :) ')