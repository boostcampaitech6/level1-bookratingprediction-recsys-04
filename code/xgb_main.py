from tqdm import tqdm
import time
import wandb
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys
# code가 있는 path로 지정해주세요. 그렇지 않으면 console에서 실행시 에러가 납니다.
sys.path.append("/data/ephemeral/home/level1-bookratingprediction-recsys-04/code")
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
from sklearn.decomposition import PCA
import xgboost as xgb

# Argument 설정 (기본값)
#data_path="/data/ephemeral/home/level1-bookratingprediction-recsys-04/data/"
parser = argparse.ArgumentParser(description='parser')
arg=args = parser.parse_args()
arg.data_path='data/'
arg.test_size = 0.2
arg.seed = 42
arg.batch_size = 1
arg.data_shuffle = True

# seed 고정
Setting.seed_everything(arg.seed)

# context data Loader
data = context_data_load(arg)
data = context_data_split(arg, data)
data = context_data_loader(arg, data)

# Image 데이터 프로세싱 (작업중...)
image_pros=False

if image_pros:
    idata = image_data_load(arg)
    idata = image_data_split(arg, idata)
    idata = image_data_loader(arg, idata)
    idata['X_train']['img_vector_flat']=idata['X_train']['img_vector'].apply(lambda x:x.flatten())
    idata['X_valid']['img_vector_flat']=idata['X_valid']['img_vector'].apply(lambda x:x.flatten())
    idata_xtrain=idata['X_train']['img_vector_flat'].apply(pd.Series)
    idata_xvalid=idata['X_valid']['img_vector_flat'].apply(pd.Series)

    pca = PCA(n_components=6)

    idata_train_pca=pca.fit_transform(idata_xtrain)
    idata_valid_pca=pca.fit_transform(idata_xvalid)

    idata_xtrain_fin=pd.DataFrame(idata_train_pca,index=idata_xtrain.index)
    idata_xvalid_fin=pd.DataFrame(idata_valid_pca,index=idata_xvalid.index)
    
    dxtrain=pd.concat((data['X_train'], idata_xtrain_fin),ignore_index=True, axis=1)
    dxvalid=pd.concat((data['X_valid'], idata_xvalid_fin),ignore_index=True, axis=1)

    dtrain=xgb.DMatrix(dxtrain, data['y_train'])
    dvalid=xgb.DMatrix(dxvalid, data['y_valid'])


# xgboost data load 형식
dtrain=xgb.DMatrix(data['X_train'], data['y_train'])
dvalid=xgb.DMatrix(data['X_valid'], data['y_valid'])

# xgboost test data load 형식
dtest=xgb.DMatrix(data['test'])

# PARAMETER 튜닝, 옆에 숫자는 기본값
# GENERAL PARM
# gbtree(기본값), gblinear, dart
param = {'booster' : 'gbtree'}
#param = {'booster' : 'dart'}

# BOOSTER PARM
# eta = lr 0.3
#param['eta'] = 0.3
# weak learn 반복수 10
#param['num_boost_around'] = 10
# 트리 최대 깊이 6
#param['max_depth'] = 6
# L2 Reg 적용 값 1
#param['lambda'] = 1
# L1 Reg 적용 값 0
#param['alpha'] = 0

# TRAIN PARM
# reg:squarederror(기본값), binary:logistic, multi:softmax, multi:softprob
param['objective'] = 'reg:squarederror'
#param['objective'] = 'binary:logistic'

#param['objective'] = 'multi:softmax'
#param['num_class'] = 11

# EVAL metric
# rmse, mae, logloss, error, merror, mlogloss, auc
param['eval_metric'] = 'rmse'

#param['eval_metric'] = 'mlogloss'


# epoch
num_round = 100

# 모델 학습
bst = xgb.train(param, dtrain, num_round, evals=[(dvalid,"valid")])

# 모델 예측
out = bst.predict(dtest)

# DART 사용시 모델 예측 (작업중...)
#out = bst.predict(dtest, iteration_range=(0, num_round))

# 모델 예측 저장
out_file_name="xgb_base.csv"

test = pd.read_csv(args.data_path + 'test_ratings.csv')
test['rating']=out
test.to_csv(out_file_name, index=False)