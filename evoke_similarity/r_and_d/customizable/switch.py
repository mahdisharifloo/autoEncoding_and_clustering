# -*- coding: utf-8 -*-

import features as fe
import json
import cv2 
import numpy as np

image_path = '/home/mahdi/Picutres/test/2.jpeg'

image = cv2.imread(image_path)
image = cv2.resize(image,image_size)

with open('conf/conf.json') as f:    
    config = json.load(f) 
with open('conf/feature_names.json') as f:    
    features_name = json.load(f)       

if conf['shape']:
    shape       = glb_ex.shape(image)    
if conf['texture']:
    textture    = glb_ex.texture(image)
if conf['color']:
    color       = glb_ex.color(image)
if conf['SIFT']:
    sift_kp , sift_des = lcl_ex.SIFT(image_path)
if conf['SURF']:
    surf_kp,surf_des = lcl_ex.SURF(image_path)
if conf['ORB']:
    orb_kp,orb_des = lcl_ex.ORB(image_path)
if conf['BRIEF']:
    brief_kp,brief_des = lcl_ex.BRIEF(image_path)
if conf['KAZE']:
    kaze_kp,kaze_des = lcl_ex.KAZE(image_path)

def hstack_maker(*args):
    features = np.hstack(args)
    return features


####################################################
##   this is the second way for feature selection ##
####################################################
def feature_selector(first, second, third, **options):
    if options.get("action") == "sum":
        print("The sum is: %d" %(first + second + third))

    if options.get("number") == "first":
        return first


def foo(first, second, third, *therest):
    print("First: %s" %(first))
    


features = fe.


shape = getattr(fe.Global_feature_extraction(),config['global_features'][0])

for i in config['global_features']:
    print(i)
    this_fea
