import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import time
import wandb
import argparse
import pandas as pd
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

from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from catboost.utils import eval_metric
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="parser")
arg = args = parser.parse_args()
arg.data_path = "data/"
data = context_data_load(arg)

# 데이터 준비 (예: X, y)

# K-Fold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 성능 기록을 위한 리스트
scores = []

i = 0

dtest = xgb.DMatrix(data["test"])
for train_index, test_index in kf.split(data["train"]):
    i += 1
    X_train, X_valid = (
        data["train"].drop("rating", axis=1).iloc[train_index],
        data["train"].drop("rating", axis=1).iloc[test_index],
    )
    y_train, y_valid = (
        data["train"]["rating"].iloc[train_index],
        data["train"]["rating"].iloc[test_index],
    )

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    # XGBoost 모델 초기화 및 훈련
    param = {"booster": "gbtree"}
    param["eta"] = 0.1
    param["max_depth"] = 9
    param["device"] = "cuda"
    # 0
    param["gamma"] = 0.3
    # 1
    param["min_child_weight"] = 15
    # L2 Reg 적용 값 1
    param["lambda"] = 0.1
    # L1 Reg 적용 값 0
    # param['alpha'] = 0

    # TRAIN PARM
    # reg:squarederror(기본값), binary:logistic, multi:softmax, multi:softprob
    param["objective"] = "reg:squarederror"

    # EVAL metric
    # rmse, mae, logloss, error, merror, mlogloss, auc
    param["eval_metric"] = "rmse"
    # epoch
    num_round = 2000
    bst = xgb.train(
        param,
        dtrain,
        num_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=100,
    )

    # 성능 평가
    score = bst.eval(dvalid)
    scores.append(float(score[14:]))

    if i == 1:
        out = {"k-fold: " + str(i): bst.predict(dtest)}
    else:
        out["k-fold: " + str(i)] = bst.predict(dtest)

test = pd.read_csv(args.data_path + "test_ratings.csv")
test["rating"] = pd.DataFrame(out).mean(axis=1)
print(np.array(scores).mean())
test["rating"][test["rating"] > 10] = 10


test.to_csv("xgb_5fold.csv", index=False)
