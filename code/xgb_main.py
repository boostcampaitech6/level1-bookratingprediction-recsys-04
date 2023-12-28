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
# data_path="/data/ephemeral/home/level1-bookratingprediction-recsys-04/data/"
parser = argparse.ArgumentParser(description="parser")
arg = args = parser.parse_args()
arg.data_path = "data/"
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
image_pros = False

if image_pros:
    idata = image_data_load(arg)
    idata = image_data_split(arg, idata)
    idata = image_data_loader(arg, idata)
    idata["X_train"]["img_vector_flat"] = idata["X_train"]["img_vector"].apply(
        lambda x: x.flatten()
    )
    idata["X_valid"]["img_vector_flat"] = idata["X_valid"]["img_vector"].apply(
        lambda x: x.flatten()
    )
    idata_xtrain = idata["X_train"]["img_vector_flat"].apply(pd.Series)
    idata_xvalid = idata["X_valid"]["img_vector_flat"].apply(pd.Series)

    pca = PCA(n_components=6)

    idata_train_pca = pca.fit_transform(idata_xtrain)
    idata_valid_pca = pca.fit_transform(idata_xvalid)

    idata_xtrain_fin = pd.DataFrame(idata_train_pca, index=idata_xtrain.index)
    idata_xvalid_fin = pd.DataFrame(idata_valid_pca, index=idata_xvalid.index)

    dxtrain = pd.concat((data["X_train"], idata_xtrain_fin), ignore_index=True, axis=1)
    dxvalid = pd.concat((data["X_valid"], idata_xvalid_fin), ignore_index=True, axis=1)

    dtrain = xgb.DMatrix(dxtrain, data["y_train"])
    dvalid = xgb.DMatrix(dxvalid, data["y_valid"])

# binary classification data
data["y_train_bin"] = data["y_train"].copy()
data["y_train_bin"][data["y_train"] >= 6] = 1
data["y_train_bin"][data["y_train"] <= 5] = 0

data["y_valid_bin"] = data["y_valid"].copy()
data["y_valid_bin"][data["y_valid"] >= 6] = 1
data["y_valid_bin"][data["y_valid"] <= 5] = 0


# xgboost data load 형식
dtrain = xgb.DMatrix(data["X_train"], data["y_train"])
dvalid = xgb.DMatrix(data["X_valid"], data["y_valid"])

dtrain_bin = xgb.DMatrix(data["X_train"], data["y_train_bin"])
dvalid_bin = xgb.DMatrix(data["X_valid"], data["y_valid_bin"])


# xgboost test data load 형식
dtest = xgb.DMatrix(data["test"])

# PARAMETER 튜닝, 옆에 숫자는 기본값
# GENERAL PARM
# gbtree(기본값), gblinear, dart
param = {"booster": "gbtree"}
param_bin = {"booster": "gbtree"}
# param = {'booster' : 'dart'}
param["device"] = "cuda"

# BOOSTER PARM
# eta = lr 0.3
param["eta"] = 0.3
param_bin["eta"] = 0.2
# weak learn 반복수 10
# param['num_boost_around'] = 10
# 트리 최대 깊이 6
param["max_depth"] = 6
param_bin["max_depth"] = 9
# L2 Reg 적용 값 1
# param['lambda'] = 1
# L1 Reg 적용 값 0
# param['alpha'] = 0

# TRAIN PARM
# reg:squarederror(기본값), binary:logistic, multi:softmax, multi:softprob
param["objective"] = "reg:squarederror"
param_bin["objective"] = "binary:logistic"

# param['objective'] = 'multi:softmax'
# param['num_class'] = 11

# EVAL metric
# rmse, mae, logloss, error, merror, mlogloss, auc
param["eval_metric"] = "rmse"
param_bin["eval_metric"] = "error"
# param['eval_metric'] = 'mlogloss'


# epoch
num_round = 400
num_round_neg = 100

# 모델 학습
bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, "train"), (dvalid, "valid")])

mod_bin = False
if mod_bin:
    bst_bin = xgb.train(
        param_bin,
        dtrain_bin,
        num_round,
        evals=[(dtrain_bin, "train"), (dvalid_bin, "valid")],
    )
    out_train_bin = bst_bin.predict(dtrain)
    out_valid_bin = bst_bin.predict(dvalid)
    out_test_bin = bst_bin.predict(dtest)

    # train set에 대해서 p=0.743 > 인 data를 pos로 치환했을때 비슷한 개수의 data가 나옴
    data["X_train"][data["y_train"] >= 6].size
    data["X_train"][out_train_bin > 0.743].size

    # valid set에 대해서 p=0.743 > 인 data를 pos로 치환했을때 비슷한 개수의 data가 나옴
    data["X_valid"][data["y_valid"] >= 6].size
    data["X_valid"][out_valid_bin > 0.743].size

    # train index 생성방법
    # 추론을 통해 index 생성
    pos_ind_train = out_train_bin > 0.743
    pos_ind_valid = out_valid_bin > 0.743

    # 명시적으로 index 생성
    # pos_ind_train = data["y_train"] >= 6
    # pos_ind_valid = data["y_valid"] >= 6

    # test index 추론
    pos_ind_test = out_test_bin > 0.743

    data_pos = {"X_train": data["X_train"][pos_ind_train]}
    data_neg = {"X_train": data["X_train"][~pos_ind_train]}

    data_pos["X_valid"] = data["X_valid"][pos_ind_valid]
    data_neg["X_valid"] = data["X_valid"][~pos_ind_valid]

    data_pos["y_train"] = data["y_train"][pos_ind_train]
    data_neg["y_train"] = data["y_train"][~pos_ind_train]
    data_pos["y_valid"] = data["y_valid"][pos_ind_valid]
    data_neg["y_valid"] = data["y_valid"][~pos_ind_valid]

    # 이중 모델링 (bin -> reg)
    postrain = xgb.DMatrix(data_pos["X_train"], data_pos["y_train"])
    posvalid = xgb.DMatrix(data_pos["X_valid"], data_pos["y_valid"])

    negtrain = xgb.DMatrix(data_neg["X_train"], data_neg["y_train"])
    negvalid = xgb.DMatrix(data_neg["X_valid"], data_neg["y_valid"])

    pos_dtest = xgb.DMatrix(data["test"][pos_ind_test])
    neg_dtest = xgb.DMatrix(data["test"][~pos_ind_test])

    dm_xgb = True
    if dm_xgb:
        pos_bst = xgb.train(
            param, postrain, num_round, evals=[(postrain, "train"), (posvalid, "valid")]
        )
        neg_bst = xgb.train(
            param,
            negtrain,
            num_round_neg,
            evals=[(negtrain, "train"), (negvalid, "valid")],
        )

        pos_out = pos_bst.predict(pos_dtest)
        neg_out = neg_bst.predict(neg_dtest)

    # Catboost 이용 neg_predict
    dm_cat = False
    if dm_cat:
        from catboost import CatBoostClassifier, Pool, CatBoostRegressor
        from catboost.utils import eval_metric
        from sklearn.datasets import make_multilabel_classification
        from sklearn.model_selection import train_test_split

        train_pool = Pool(data_neg["X_train"], data_neg["y_train"])
        test_pool = Pool(data_neg["X_valid"], data_neg["y_valid"])

        clf = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE", iterations=500
        )
        clf.fit(train_pool, eval_set=test_pool, metric_period=10, verbose=50)

        neg_out = clf.predict(data["test"])

out = bst.predict(dtest)

# cat_model = CatBoostRegressor(task_type='GPU', devices='0')
# cat_model.fit(train_pool, eval_set=test_pool)

# DART 사용시 모델 예측 (작업중...)
# out = bst.predict(dtest, iteration_range=(0, num_round))

# 모델 예측 저장
out_file_name = "xgb.csv"

test = pd.read_csv(args.data_path + "test_ratings.csv")
test["rating"] = test["rating"].astype("float32")

#
test["rating"] = out

#
# test["rating"][pos_ind_test] = pos_out
# test["rating"][~pos_ind_test] = neg_out
test["rating"]

test.to_csv(out_file_name, index=False)
