from unet_model import *
from data import *

TRAINPATH = '/Users/suhaas/Desktop/stott_lab/HN_Segmentation'

aug_gen_param = dict(rescale=1./255, 
					rotation_range=2,
					width_shift_range=0.05,
					height_shift_range=0.05, 
					shear_range=0.05,
					zoom_range=0.05,
					horizontal_flip=True, 
					vertical_flip=True,
					fill_mode='reflect')

training_gen = trainGenerator(5, TRAINPATH, 'image', 'mask', aug_gen_param, save_to_dir = None)

model = unet()
model.fit_generator(training_gen, steps_per_epoch=300, epochs=1)

