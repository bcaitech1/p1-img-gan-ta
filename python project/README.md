## 코드에 대한 전반적인 설명

### 1, dataloader
mask.py:데이터 셋에 관련된 클래스 및 함수를 정의

MaskBaseDataset : 기본적인 데이터 셋에 대한 클래스

MultiHeadMaskDataset : multi head classification을 위한 데이터 셋 클래스

MaskBaseDatasetFold : k-fold cross validation을 하기 위한 데이터 셋 클래스

TestDataset : 테스트 데이터셋을 위한 데이터셋 클래스

TestDatasetTTA : TTA inference를 위한 테스트 데이터 셋 클래스

### 2, inference , inference_notebook
test데이터를 추론하기 위한 코드들을 모아놨습니다.

이때, 주로 노트북 파일을 사용하였고 앙상블 기법을 다양하게 적용해 보기 위해 그때그때마다 수정해서 사용을
하였습니다.

### 3, model
model.py: pretrained 된 모델을 커스터마이징하여 사용할 수 있게 클래스로 만들어 두웠습니다.

metric.py: f1 score를 계산하기 위한 함수를 정의 해 두웠습니다.

loss.py : f1 loss, label smooting, focal loss 함수들을 정의 해 두웠습니다.

### 4, train
<X_... : 이전에 사용했다가 사용하지 않는 파일>

X_train_calss1up.py  : 1번 클래스의 데이터만 증가시켜 학습을 진행합니다.

X_train-fold-class-balance.py : 클래스별 균형을 맞춰 k-fold cross validation방법으로 학습을 진행합니다.

X_train -fold-people-balance.py : 사람을 기준으로 균형을 맞춰 k-fold cross validation방법으로 학습을 진행합니다.

noval.py : validation set없이 모든 train데이터를 사용하여 학습을 진행합니다.

multi.py : multihead classification방법으로 학습을 진행합니다.

fold.py : k-fold cross validation방법으로 학습을 진행합니다.

normal.py : 9:1로 데이터 셋을 쪼개어 train, valid로 진행합니다.

train.py : noval.py, multi.py, fold.py, normal.py를 사용하여 학습을 맞게 진행하고 필요한 파라미터는
json파일 형식으로 읽어와 진행합니다.

### 5, util
generate : cutmix, mixup를 수행하는 함수를 가지고 있습니다

meter : train시 loss, acc계산을 도와주는 클래스를 가지고 있습니다.

transformers :Data Augmentation에 대한 정보를 가지고 파라미터에 맞는 알맞는 Augmentation기법을 반환해줍니다.