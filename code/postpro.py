import pandas as pd

data_path='/data/ephemeral/home/level1-bookratingprediction-recsys-04/code/submit/'
file_name='aw-xgb_base-xgb_mc-CNN-FM-DCN-DeepCoNN-FFM.csv'


test = pd.read_csv(data_path + file_name)
#print(test)
test.loc[test['rating'] > 10,'rating'] = 10


test[test['rating'] > 10]
test[test['rating'] < 1]

test.to_csv("out_post.csv", index=False)