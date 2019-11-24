# 필요한 패키지들 
import os 
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array 
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys

#tf.debugging.set_log_device_placement(True)

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)
"""

def get_label_from_path(path):
    return int(path.split('/')[-2])

# HEAD : Width  Height  Roi.X1  Roi.Y1  Roi.X2  Roi.Y2  ClassId  Path
train_csv = pd.read_csv('data/gtsrb-german-traffic-sign/Train.csv')
test_csv = pd.read_csv('data/gtsrb-german-traffic-sign/Test.csv')

train_list = train_csv['Path']  # 모든 경로들을 list로 반환
test_list = test_csv['Path']

#print(train_csv.iloc[1]['Width'])
base_data_path = 'data/gtsrb-german-traffic-sign'
train_label_name_list = []
test_label_name_list = []

train_label_name_list = train_csv['ClassId']
test_label_name_list = test_csv['ClassId']

unique_label_names = np.unique(test_label_name_list)

# Hyper Parameter 
batch_size = 64
data_height = 28
data_width = 28
channel_n = 3
num_classes = len(unique_label_names)

# 방법.1 - Empty Array를 만들고 채워가는 방법
train_images = np.zeros((len(train_list), data_height, data_width, channel_n))
train_labels = np.array(train_label_name_list)

test_images = np.zeros((len(test_list), data_height, data_width, channel_n))
test_labels = np.array(test_label_name_list)

def read_image(path):
    path = base_data_path + '/' + path
    image = Image.open(path).resize((data_height, data_width))
    image = np.array(image.convert('L'))
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(data_height, data_width, 1)

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
    

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(data_height, data_width, channel_n)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(43, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_loss, test_acc)
