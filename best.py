# 필요한 패키지들 
import os 
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array 
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

import sys
import pickle
import gzip

#tf.debugging.set_log_device_placement(True)
#tf.config.gpu_options.allow_growth = True

def init_gpu():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
		except RuntimeError as e:
			print(e)

# Hyper Parameter 
batch_size = 64
data_height = 32
data_width = 32
channel_n = 3
num_classes = 43

#Randomize the order of the input images
#s = np.arange(train_images.shape[0])
#np.random.shuffle(s)
#train_images = train_images[s]
#train_labels = train_labels[s]

#s = np.arange(test_images.shape[0])
#np.random.shuffle(s)
#test_images = test_images[s]
#test_labels = test_labels[s]

def data():
	with gzip.open('pickle/train_images.pickle', 'rb') as f:
		train_images = pickle.load(f)

	with gzip.open('pickle/train_labels.pickle', 'rb') as f:
		train_labels = pickle.load(f)

	with gzip.open('pickle/test_images.pickle', 'rb') as f:
		test_images = pickle.load(f)

	with gzip.open('pickle/test_labels.pickle', 'rb') as f:
		test_labels = pickle.load(f)


	return train_images, train_labels, test_images, test_labels

def create_model(train_images, train_labels, test_images, test_labels):
	model = models.Sequential()
	
	model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(data_width, data_height, channel_n)))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(120, activation='relu'))
	model.add(Dense(84, activation='relu'))
	
	model.add(Dense(43, activation='softmax'))
	
	return model

if __name__ == '__main__':

	init_gpu()
	

	X_train, Y_train, X_test, Y_test = data()
	model = create_model(X_train, Y_train, X_test, Y_test)

	model.summary()

	model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
			metrics=["accuracy"])
	
	model.fit(X_train, Y_train, epochs=10, batch_size=64)
	
	test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
	print(test_loss, test_acc)	
