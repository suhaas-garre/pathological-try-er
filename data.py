
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import cv2
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import glob
import random

IMG_WIDTH = 1340
IMG_HEIGHT = 1004
BATCH_SIZE = 5

Tumor = []
Lymphocytes = []
Macrophages = []
Stroma = []
Epith = []

COLOR_DICT = np.array([Tumor, Lymphocytes, Macrophages, Stroma, Epith])
