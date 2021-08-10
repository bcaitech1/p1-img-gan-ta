# 마스크 착용 상태 분류 Project(Boostcamp P stage1)
본프로젝트는 NAVER AI BoostCamp에서 개최한 competition입니다

## 최종 결과
**`Public f1 Score`**  0.7784

**`Private f1 Score`** 0.7632

**`Private Rank`** 14/220

## 대회 목표
카메라로 비친 사람 얼굴 이미지를 이용하여 마스크 착용 여부, 나이, 연령대별로 18개의 클래스로 이미지를 분류

## 문제 해결법

- 해당 문제를 풀기 위해 어떻게 접근하고, 구현하고, 최종적으로 사용한 솔루션에 대해서는 [report](https://www.notion.so/P-Stage1-Wrap-UP-5c862021777243dd9ac1befebe345926)에서 확인할 수 있습니다

- 위 Report는 개인의 회고도 포함하고 있습니다.

## 전체 진행 프로세스

![](./image/process.png)


## 📜 최종 모델 및 파라미터
   1. model
        1. resnext101_32x8d
   2. Learning Rate: 3e-4 
   3. Optimizer : Adam
   4. Loss : focal loss
   5. Epoch: 4
   6. Scheduler : MultiStepLR milestones
   7. batch_size : 16

## 🌟 문제 해결 측면
### **[EDA]**
- matplotlib, seaborn라이브러리 활용

- 나이에 대한 분포도 분석 - Scatter plot

- 나이별, 성별, 마스크 상태의 분포도 분석 - barplot

- 클래스별 분포도 분석 - pieplot

- Nan값의 유무 파악 - barplot

### **[Augmentation]**
- CenterCrop: 사람의 얼굴과 목 부분이 나오게끔 이미지를 잘랐습니다.

- horizontal flip: 뒤집힌 사진은 존재하지 않기에 좌우 대칭을 주는 것만 사용하는 것이 옳다고 판단하였습니다.

- CLAHE: 사람의 나이를 판별하는데는 주름같은 윤곽선이나 피부의 상태를 좀 더 강하게 주는 것이 도움이 된다고 판단하여 사용하기로 결정히였습니다.

- RandomErase: 한곳에만 모델이 집중되는 현상을 막고 성능향상의 결과로 이어졌다는 논문 내용을 보고 마스크에 weight들이 집중되는 것을 분산시키고자 사용하였습니다. 

- mixup, cutmix: 모델에 어려운 테스크를 줌으로써 성능을 높히고자 하였습니다.

- CenterCrop, CLAHE Augmentation을 사용하여 성능 향상을 시켰습니다.

### **[model]**
사용 모델
- VGG19
- GoogleNet
- ResNet18
- EfficientNet b1
- DenseNet
- ResNext(최종 모델)

### **[class imbalance, Data label]**
1. hrad way
   1. oversampling
   2. undersampling<br>
   => 해당 대회에서는 train데이터의 수가 그리 많지 않은 편이고 불균형이 심하였으며 downsampling을 하기 위해서는 특징을 잘 나타내는 사진만 골라야 하는 여러 문제가 있었기에 oversampling을 사용하기로 결정

2. SoftWay
    1. focal loss
    2. f1 loss
    3. label smooting
    
### **[실험환경 구축]**
- 파이썬 코드 모듈화 및 wandb적용

- config파일을 이용한 모델 설정 간편화

### **[모델분석 및 해결하기 위한 노력]**
- class1에 대한 정확도 낮은 모델의 상태 분석

<img src = "https://user-images.githubusercontent.com/51118441/120631706-1f5f8880-c4a3-11eb-857c-5b3dbe04f0bc.png" width="70%">

- 해결하기 위한 노력<br>1. class1에 대해서만 MixUp(2번 클래스와 mixup을 수행)하고  class1에 대해서만 upsampling하여 학습을 시킴
<br>2.class1에 대해서만 cutmix(2번 클래스와 cutmix를 수행)하고 class1에 대해서만 upsampling하여 학습을 시켜봄 학습을 시킴
<br>3.외부 데이터를 사용하고 class1에 대해서만 upsampling하여 학습을 시킴
<br>4.기존 성능이 좋은 모델 + 1번 class만 upsampling하여 앙상블 시도
<br>=> 모델에 대해서 해당 클래스에 대하여 어려운 테스크를 수행시키거나 데이터의 수를 늘려 학습시키는 방향의 시도를 하였습니다.

### **[이외의 시도]**
- Multihead classification(마지막 classification부분에 나이, 성별, 마스크 3가지 형태로 구분하는 3개의 linear층을 추가해 학습 수행)

- TTA(Test time Augmentation)

- K-fold cross validation

- Ensemble(단순 평균)

