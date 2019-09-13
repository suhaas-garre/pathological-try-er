import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def dice_coef(y_true, y_pred, smooth=1):
	y_true_f = keras.flatten(y_true)
	y_pred_f = keras.flatten(y_pred)
	intersection = keras.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)


def jaccard_coef(y_true, y_pred, smooth=1):
	intersection = keras.sum(y_true * y_pred)
	sum_ = keras.sum(y_true + y_pred)
	return (intersection + smooth) / (sum_ - intersection + smooth)


def unet(pretrained_weights = None, input_size = (512, 512, 7)):
	inputs = Input(input_size)
	conv1 = Conv2D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = Conv2D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

	conv2 = Conv2D(128, 128, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 128, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

	conv3 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

	conv4 = Conv2D(512, 512, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 512, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

	conv5 = Conv2D(1024, 1024, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	conv5 = Conv2D(1024, 1024, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

	up6 = concatenate([Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(512, 512, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
	conv6 = Conv2D(512, 512, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
	conv7 = Conv2D(256, 256, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(128, 128, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
	conv8 = Conv2D(128, 128, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
	conv9 = Conv2D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

	conv10 = Conv2D(1, (1,1), activation = 'sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

	return model