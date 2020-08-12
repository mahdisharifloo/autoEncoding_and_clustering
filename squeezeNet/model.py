# -*- coding: utf-8 -*-

import numpy as np
from keras_squeezenet import SqueezeNet
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = SqueezeNet()


import squeezenet

model = squeezenet.SqueezeNet()
model.summary()