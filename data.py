
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

Tumor = []
Lymphocytes = []
Macrophages = []
Stroma = []
Epith = []

COLOR_DICT = np.array([Tumor, Lymphocytes, Macrophages, Stroma, Epith])

def trainGenerator(batch_size, train_path,image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                     mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                     flag_multi_class = False, num_class = 2, save_to_dir = None, target_size = (512,512), seed = 1):

	img_datagen = ImageDataGenerator(**aug_dict)
	# Should I fit to x_train here? 
	mask_datagen = ImageDataGenerator(**aug_dict)

	img_generator = img_datagen.flow_from_directory(
		train_path,
		classes = [image_folder], 
		class_mode = None,
		color_mode = image_color_mode,
		target_size = target_size, 
		batch_size = batch_size, 
		save_to_dir = save_to_dir, 
		save_prefix = image_save_prefix,
		seed = seed)
	mask_generator = mask_datagen.flow_from_directory(
		train_path,
		classes = [mask_folder], 
		class_mode = None,
		color_mode = mask_color_mode,
		target_size = target_size, 
		batch_size = batch_size, 
		save_to_dir = save_to_dir, 
		save_prefix = mask_save_prefix,
		seed = seed)
	
	train_generator = zip(img_generator, mask_generator)
	for (img, mask) in train_generator:
		img, mask = adjustData(img, mask, flag_multi_class, num_class)
		yield (img, mask)
