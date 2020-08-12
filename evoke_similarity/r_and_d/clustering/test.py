import cv2
import numpy as np

# Local dependencies
import utils
import constants

orb = cv2.ORB()
kp, des = orb.detectAndCompute(img, None)