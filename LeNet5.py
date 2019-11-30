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

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

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
data_height = 28 
data_width = 28
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
		
	model_choice = {{choice(['one', 'two'])}}

	if model_choice == 'one':
		model.add(Conv2D(filters=32, kernel_size={{choice([3, 5])}}, activation='relu', input_shape=(32, 32, 3)))
		model.add(MaxPool2D(pool_size=(2, 2)))
		model.add(Conv2D(filters=64, kernel_size={{choice([3, 5])}}, activation='relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))

	elif model_choice == 'two':	
		model.add(Conv2D(filters=64, kernel_size={{choice([3, 5])}}, activation='relu', input_shape=(32, 32, 3)))
		model.add(MaxPool2D(pool_size=(2, 2)))
		model.add(Conv2D(filters=128, kernel_size={{choice([3, 5])}}, activation='relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense({{choice([256, 512,1024])}}, activation='relu'))

	model_choice = {{choice(['one', 'two'])}}
	
	if model_choice == 'two':
		model.add(Dense({{choice([256, 512,1024])}}, activation='relu'))
	
	model.add(Dense(43, activation='softmax'))
	
	model.compile(loss="sparse_categorical_crossentropy", optimizer={{choice(['adam', 'sgd'])}},
			metrics=["accuracy"])
	
	result = model.fit(train_images, train_labels, epochs=5, batch_size={{choice([32, 64])}})

	#get the highest validation accuracy of the training epochs
	validation_acc = np.amax(result.history['acc']) 
	print('Best validation acc of epoch:', validation_acc)

	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

	init_gpu()
	
	best_run, best_model = optim.minimize(model=create_model,
			data=data,
			algo=tpe.suggest,
			max_evals=10,
			trials=Trials())

	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)
