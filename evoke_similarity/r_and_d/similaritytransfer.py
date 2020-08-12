#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:25:17 2019

@author: reyhane
"""
from keras.preprocessing import 

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

IMAGE_SIZE = [224, 224] # feel free to change depending on dataset

# training config:
epochs = 10
batch_size = 32

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = 'train'
valid_path = 'valid'

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')

res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in res.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(res.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=res.input, outputs=prediction)
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

train_generator = gen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = gen.flow_from_directory(
    valid_path,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical')
#train  model
r = model.fit_generator(
  train_generator,
  validation_data=validation_generator,
  epochs=10,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)



from sklearn.metrics.pairwise import cosine_similarity 
feat_extract = Model(inputs=model.input,outputs=model.layers[-2].output)
imgs_features = feat_extract.predict(prediction)
cosSimilarities = cosine_similarity(imgs_features)

class ImageRecommender : 
    
    def __init__(self, model, list_of_image, filespath) : 
        self.model = model
        self.filespath = filespath
        self.list_of_image = list_of_image
        #since ouput.shape return object dimension just eval it to get integer ...
        self.image_width = eval(str(self.model.layers[0].output.shape[1]))
        self.image_height = eval(str(self.model.layers[0].output.shape[2]))
        # remove the last layers in order to get features instead of predictions
        self.image_features_extractor = Model(inputs=self.model.input, 
                                              outputs=self.model.layers[-2].output)
        self.processed_image = self.Pics2Matrix()
        self.sim_table = self.GetSimilarity(self.processed_image)
        
    def ddl_images(self, image_url) :
        try : 
            return load_img(self.filespath + image_url, 
                            target_size=(self.image_width, self.image_height))
        except OSError : 
            # image unreadable // remove from list
            self.list_of_image = [x for x in self.list_of_image if x != image_url]
            #self.list_of_image.remove(image_url)
            pass
        
    def Pics2Matrix(self) :
        """
        # convert the PIL image to a numpy array
        # in PIL - image is in (width, height, channel)
        # in Numpy - image is in (height, width, channel)
        # convert the image / images into batch format
        # expand_dims will add an extra dimension to the data at a particular axis
        # we want the input matrix to the network to be of the form (batchsize, height, width, channels)
        # thus we add the extra dimension to the axis 0.
        """
        #from keras.preprocessing.image import load_img,img_to_array
        list_of_expanded_array = list()
        for i in tqdm(range(len(self.list_of_image) - 1)) :
            try :
                tmp = img_to_array(self.ddl_images(self.list_of_image[i]))
                expand = np.expand_dims(tmp, axis = 0)
                list_of_expanded_array.append(expand)
            except ValueError : 
                self.list_of_image = [x for x in self.list_of_image if x != self.list_of_image[i]]
                #self.list_of_image.remove(self.list_of_image[i])
        images = np.vstack(list_of_expanded_array)
        """
        list_of_expanded_array = [try np.expand_dims(img_to_array(self.ddl_images(self.list_of_image[i])), axis = 0) except ValueError pass \
                                  for i in tqdm(range(len(self.list_of_image)))]
        images = np.vstack(list_of_expanded_array)
        #from keras.applications.imagenet_utils import preprocess_input()
        # prepare the image for the  model"
        """
        return preprocess_input(images)
    
    def GetSimilarity(self, processed_imgs) :
        print('============ algorithm predict featurs =========')
        imgs_features = self.image_features_extractor.predict(processed_imgs)
        print("Our image has %i features:" %imgs_features.size)
        cosSimilarities = cosine_similarity(imgs_features)
        cos_similarities_df = pd.DataFrame(cosSimilarities, 
                                           columns=self.list_of_image[:len(self.list_of_image) -1],
                                           index=self.list_of_image[:len(self.list_of_image) -1])
        return cos_similarities_df
    
    def most_similar_to(self, given_img, nb_closest_images = 5):

        print("-----------------------------------------------------------------------")
        print("original manga:")

        original = self.ddl_images(given_img)
        plt.imshow(original)
        plt.show()

        print("-----------------------------------------------------------------------")
        print("most similar manga:")

        closest_imgs = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
        closest_imgs_scores = self.sim_table[given_img].sort_values(ascending=False)[1:nb_closest_images+1]

        for i in range(0,len(closest_imgs)):
            original = self.ddl_images(closest_imgs[i])
            plt.imshow(original)
            plt.show()
            print("similarity score : ",closest_imgs_scores[i])