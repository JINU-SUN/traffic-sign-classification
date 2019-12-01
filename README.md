# traffic-sign-classification

## Dataset
gtsrdb 데이터셋을 사용하였다. 해당 데이터는 아래 링크를 통해 얻을 수 있다. 
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

해당 데이터을 받았으면,  data directory 를 만들고 해당 directory에 압축을 푼다.

빠른 Train을 위해 preprocessing 데이터 구조를 pickle에 저장
```bash
mkdir pickle
python create_pickle.py
```

## Pretrained Model
미리 학습한 모델은 model directory에 존재하며 inferences.ipynb 를 통해 해당 모델을 load해서 사용한다.

## Train
위에서 받은 Dataset을 가지고 train.ipynb를 통해 학습한다.
