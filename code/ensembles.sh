#!/bin/bash
##########

# 앙상블 할 filename 순서대로 .csv는 필요없음
###Example###
###filenames=xgb_baseline
###filenames=xgb_baseline,20231213_085959_CNN_FM,dcn,20231215_040951_DeepCoNN

filenames=xgb_base,xgb_mc,CNN-FM,DCN,DeepCoNN,FFM

# 앙상블 전략
# 가설: boost 기반: 잔차를 보정하기 때문에 과적합이 심함 반대로 말하면 개인의 특징을 잘 잡아냄
#       DL 기반: 전체적인 데이터의 특성을 학습, 일반화 성능이 뛰어남
# rating = boost rating + DL rating으로 나누어서 계산, boost rating은 각각의 특징을 잘 잡아낸다고 가정
# DL rating은 전체적인 데이터의 특성을 잘 보정한다고 가정
# 각각의 rating 계산식
# boost rating = rmse를 weight로 한 boost model output 간의 weight 보정
# DL rating = rmse를 weight로 한 DL model output 간의 weight 보정

# 앙상블 전략 'div', 'basic'중 선택
ens_opt='div'

### Model RMSE
# 모델의 valid RMSE를 작성 (seed:42 고정)
# CASE: ens_opt == div
# Boost 모델의 rmse
ens_brmse='2.27468,1.98615,2.470'
# DL 모델의 rmse
ens_drmse='2.396,2.187,2.339'


# CASE: ens_opt == basic
# 모델 rmse를 순서대로 작성
ens_rmse='2.27468,1.98615,2.396,2.187,2.339,2.470'

python rmse2w.py --b_rmse ${ens_brmse} --d_rmse ${ens_drmse} --rmse ${ens_rmse} --esb_opt ${ens_opt}


#ens_st=weighted
#ens_w='0.16588831,0.18998707,0.15748866,0.17253902,0.16132656,0.15277038'
#python ./ensemble.py --ensemble_files ${filenames} --ensemble_strategy ${ens_st} --ensemble_weight ${ens_w}


ens_st=mixed
python ./ensemble.py --ensemble_files ${filenames}




