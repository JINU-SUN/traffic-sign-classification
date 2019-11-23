# 필요한 패키지들 
import os 
from glob import glob
# PIL는 이미지를 load 할 때, numpy는 array 
from PIL import Image
import numpy as np
import pandas as pd

def get_label_from_path(path):
    return int(path.split('/')[-2])

# HEAD : Width  Height  Roi.X1  Roi.Y1  Roi.X2  Roi.Y2  ClassId  Path
train_csv = pd.read_csv('data/gtsrb-german-traffic-sign/Train.csv')

train_list = train_csv['Path']  # 모든 경로들을 list로 반환

#print(train_csv.iloc[1]['Width'])
base_data_path = 'data/gtsrb-german-traffic-sign'
label_name_list = []

for path in train_list:
    label_name_list.append(get_label_from_path(path))
    
unique_label_names = np.unique(label_name_list)


def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)

    return onehot_label

# Hyper Parameter 
batch_size = 64
data_height = 28
data_width = 28
channel_n = 1
num_classes = len(unique_label_names)

# 방법.1 - Empty Array를 만들고 채워가는 방법
batch_image = np.zeros((batch_size, data_height, data_width, channel_n))
batch_label = np.zeros((batch_size, num_classes))

def extract_image(image, roi_x1, roi_y1, roi_x2, roi_y2):
    image = image[roi_y1:roi_y2, roi_x1:roi_x2]
    return image

def read_image(path):
    path = base_data_path + '/' + path
    image = Image.open(path).resize((data_height, data_width))
    image = np.array(image.convert('L'))
    # Channel 1을 살려주기 위해 reshape 해줌
    return image.reshape(data_height, data_width, 1)

# 간단한 batch data 만들기
for n, path in enumerate(train_list[:batch_size]):
    row = train_csv.iloc[n]

    roi_x1 = row['Roi.X1']
    roi_x2 = row['Roi.X2']
    roi_y1 = row['Roi.Y1']
    roi_y2 = row['Roi.Y2']

    full_path = base_data_path + '/' + path
    
    image = Image.open(full_path)
    image = image.convert('L')
    image = image.crop((roi_x1, roi_y1, roi_x2, roi_y2))
    image = image.resize((data_height, data_width))
    image = np.array(image)
    image = image.reshape(data_height, data_width, 1)

    onehot_label = onehot_encode_label(path)
    batch_image[n, :, :, :] = image
    batch_label[n, :] = onehot_label

print(batch_image.shape, batch_label.shape)

