# -*- coding: utf-8 -*-


from macher import Matcher
import cv2
from matplotlib import pyplot as plt






def show_img(image_path,image_size=tuple((500, 500))):
    image = cv2.imread(image_path)
    image = cv2.resize(image,image_size)
    plt.imshow(image)
    plt.show()