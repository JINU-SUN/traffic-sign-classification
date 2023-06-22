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
import pickle
import gzip
import sys


# HEAD : Width  Height  Roi.X1  Roi.Y1  Roi.X2  Roi.Y2  ClassId  Path
train_csv = pd.read_csv('data/Train.csv')
test_csv = pd.read_csv('data/Test.csv')

train_list = train_csv['Path']  # 모든 경로들을 list로 반환
test_list = test_csv['Path']

#print(train_csv.iloc[1]['Width'])
base_data_path = 'data/'
train_label_name_list = []
test_label_name_list = []

train_label_name_list = train_csv['ClassId']
test_label_name_list = test_csv['ClassId']

unique_label_names = np.unique(test_label_name_list)

# Hyper Parameter 
data_height = 32 
data_width = 32
channel_n = 3
num_classes = len(unique_label_names)

# 방법.1 - Empty Array를 만들고 채워가는 방법
train_images = np.zeros((len(train_list), data_height, data_width, channel_n))
train_labels = np.array(train_label_name_list)

test_images = np.zeros((len(test_list), data_height, data_width, channel_n))
test_labels = np.array(test_label_name_list)

# 간단한 batch data 만들기
for n, path in enumerate(train_list):
    row = train_csv.iloc[n]

    roi_x1 = row['Roi.X1']
    roi_x2 = row['Roi.X2']
    roi_y1 = row['Roi.Y1']
    roi_y2 = row['Roi.Y2']

    full_path = base_data_path + '/' + path
    
    image = Image.open(full_path)
    image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
    image = image.resize((data_height, data_width))
    image = np.array(image)
    image = image.reshape(data_height, data_width, channel_n)

    train_images[n, :, :, :] = image

for n, path in enumerate(test_list):
    row = test_csv.iloc[n]

    roi_x1 = row['Roi.X1']
    roi_x2 = row['Roi.X2']
    roi_y1 = row['Roi.Y1']
    roi_y2 = row['Roi.Y2']

    full_path = base_data_path + '/' + path
    
    image = Image.open(full_path)
    image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
    image = image.resize((data_height, data_width))

    image = np.array(image)
    image = image.reshape(data_height, data_width, channel_n)

    test_images[n, :, :, :] = image

train_images = train_images.astype('float32')/255 
test_images = test_images.astype('float32')/255 

with gzip.open('pickle/train_images.pickle', 'wb') as f:
    pickle.dump(train_images, f)

with gzip.open('pickle/train_labels.pickle', 'wb') as f:
    pickle.dump(train_labels, f)

with gzip.open('pickle/test_images.pickle', 'wb') as f:
    pickle.dump(test_images, f)

with gzip.open('pickle/test_labels.pickle', 'wb') as f:
    pickle.dump(test_labels, f)
