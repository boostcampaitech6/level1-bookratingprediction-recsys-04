# Book Rating Prediction
<br>

## 요약
본 프로젝트는 사용자와 아이템의 상호작용 정보와 메타 데이터를 활용하여 소비자의 책 평점 예측을 위한 효과적인 모델을 제작하는 것에 목적이 있다. 다양한 유형의 데이터를 활용하기 위해 FFM, DCN, DeepCoNN, ROP-CNN, CNN_FM, Catboost, XGBoost과 같은 여러 모델들을 사용하였다. 이 모델들의 장점을 취합하기 위해 이를 앙상블하여 최종적인 결과를 도출하였다. 본 프로젝트의 결과 catboost : DCN : others를 7:2:1의 비율로 앙상블한 모델의 Test RMSE가 2.14로 가장 성능이 높았으며 최종적으로 이를 제출하였다.
<br>

## 개요
뉴스기사나 짧은 러닝 타임의 동영상처럼 간결하게 콘텐츠를 즐길 수 있는 ‘숏폼 콘텐츠’에 비해 소비자들이 부담 없이 쉽게 선택할 수 있지만, 책은 완독을 위해 보다 긴 물리적인 시간이 필요하다. 또한 제목, 저자, 표지, 카테고리 등 한정된 정보로 내용을 유추하고 구매를 결정해야 하므로 선택에 더욱 신중을 가하게 된다. 우리는 소비자들의 책 구매 결정에 대한 도움을 주고자 책에 대한 메타 데이터(books), 고객에 대한 메타 데이터(users), 고객이 책에 남긴 평점 데이터(ratings)를 활용해, 1과 10 사이 평점을 예측하는 모델을 구축하고자 한다.  
<br>

## 개발 환경
python == 3.10
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia pip install -r requirement.txt
```
<br>

## 팀 구성 및 역할
<p align="center"><img src="https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-04/assets/143887451/dcbf6598-3230-40f8-bc64-cca24ca2a921" width="700"/></p>
<br>

## 프로젝트 수행 절차
<p align="center"><img src="https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-04/assets/43164670/b33daf55-d344-47c8-86e0-da13383e2481" width="800"/></p>
<br>

## 실행 방법
#### 개별 모델
- FFM, DCN, CNN-FM, DeepCoNN, ROP-CNN
```
python main.py --model model_name
```
- CatBoost
```
python Catboost.py
```
- XGBoost
```
python xgb_main.py
```
- AutoML
```
python automl.py
```

#### 앙상블
```
python ensemble.py --ensemble_files file_names
```
<br>

## 실험 결과

#### RMSE
<p align="center"><img src="https://github.com/boostcampaitech6/level1-bookratingprediction-recsys-04/assets/43164670/9dd2ffc5-3403-41da-9be6-55be6a14d233" width="1000"/></p>  
<br>

#### Catboost, DCN-par ensemble (weight: 0.7, 0.3)
- 전체적으로 성능이 뛰어났던 Catboost를 Hyperparameter 튜닝을 통해 최적화하고 이를 보완해줄 수 있는 DCN-par 모델을 같이 이용하여 ensemble을 진행하였다.
- Test RMSE 결과는 public 2.14, private 2.133으로 가장 좋은 성능을 보였다.
<br>

#### Catboost, DCN-par, Others(XGboost, H2OAutoML, CNN-FM, DeepCoNN, ROP_CNN) (weight: 0.7, 0.2, 0.1)
- 가능한 여러가지 모델의 결과를 ensemble하여 모델의 장단점을 보완하려는 시도를 하였다. 각각 모델은 Hyperparameter 튜닝을 통해 최적화된 모델이며, 머신러닝 기반 모델 부터 이미지, 텍스트를 사용하는 모델들을 모두 취합하였다.
- Test RMSE 결과는 public 2.1408, private 2.1342로 나타났다.  
