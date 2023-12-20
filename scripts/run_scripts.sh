#!/bin/bash

####USER INPUT
# BASIC OPTION

#--model : 학습 및 예측할 모델을 선택할 수 있습니다.
#		ㄴ{FM,FFM,NCF,WDN,DCN,CNN_FM,DeepCoNN} 중 하나를 선택해야 합니다.

## single run
model_multi=False
model_name=FM
## multi run
#model_multi=True
#model_name=(FM FFM NCF WDN DCN CNN_FM)


#case_name=CNN_FM_1
entity=hhun

#--data_shuffle : 데이터 셔플 여부를 조정할 수 있습니다.
#--test_size : Train/Valid split 비율을 조정할 수 있습니다.
#--seed : seed 값을 조정할 수 있습니다.
#--use_best_model : 검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.

# TRAINING OPTION
#  --batch_size : Batch size를 조정할 수 있습니다.
#  --epochs : Epoch 수를 조정할 수 있습니다.
#  --lr : Learning Rate를 조정할 수 있습니다.
#  --loss_fn : 손실 함수를 변경할 수 있습니다.
#		ㄴ {MSE,RMSE} 중 하나를 선택해야 합니다.
#  --optimizer : 최적화 함수를 변경할 수 있습니다.
#		ㄴ {SGD,ADAM} 중 하나를 선택해야 합니다.
#  --weight_decay : Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.
# GPU
#  --device : 학습에 사용할 Device를 조정할 수 있습니다.
#		ㄴ  {cuda,cpu} 중 하나를 선택해야 합니다.

#### OPTIONAL
#--saved_model_path : Saved Model path를 설정할 수 있습니다.
saved_model_path=../out/${casename}

####################################################################
#### RUNNING ####
ln -sf ../data ./
export WANDB_START_METHOD=thread

if [ $model_multi = True ] ; then
    for var in "${model_name[@]}"; do
        echo ${var} "model running..."
        echo ${var} "model running..."
        echo ${var} "model running..."
        python ../code/main.py --model ${var} --entity ${entity}
        sleep 10
    done
else
    python ../code/main.py --model ${model_name} --entity ${entity}
fi