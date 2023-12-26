import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import tqdm
import pdb
from scipy.sparse import csr_matrix, linalg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder # id2idx에 필요
from sklearn.metrics import r2_score

import random
random.seed(42)
import os
import re


from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error

warnings.filterwarnings(action='ignore')


path='../data/'    # feat siyun - 디렉토리 해당 파일에 맞게 변경 필요

users = pd.read_csv(path+'users.csv')
books = pd.read_csv(path+'books.csv')
train_ratings = pd.read_csv(path+'train_ratings.csv')
test_ratings = pd.read_csv(path+'test_ratings.csv')
submit = pd.read_csv(path + 'sample_submission.csv')


def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


#  보통 seed는 42로 설정
import random
random.seed(42)

# 멘토님의 조언에 따라 비정형 데이터 제거
books.drop(['summary', 'img_path', 'img_url'], axis = 1, inplace = True)


# isbn language 처리

# !pip install isbnlib
import isbnlib
from tqdm import tqdm


# isbn '-' 으로 구분해 새롭게 만든 'mask_isbn' 컬럼에 넣기
books['mask_isbn'] = np.nan
for i in tqdm(range(books.shape[0])):
  try:
    books.loc[i, 'mask_isbn'] = isbnlib.mask(books['isbn'][i])
  except:
    pass


# '-'로 구분한 값 중 첫 번째, 'group' 속성 추출해 새롭게 만든 'lang_isbn' 컬럼에 넣기
books['lang_isbn'] = np.nan
for i in tqdm(range(books.shape[0])):
  try:
    books.loc[i, 'lang_isbn'] = books['mask_isbn'][i].split('-')[0]
  except:
    pass


# lang_isbn 값 유 & language 무 => lang_isbn 값 유 & language 유 를 value_counts() 하여 가장 많은 값 넣기
lang_null = books.loc[books['language'].isna()]['lang_isbn'].unique()
for i in lang_null:
  try:
    candidate_lang = books.loc[books['lang_isbn']==i]['language'].value_counts().index[0]
    books.loc[books[(books['lang_isbn']==i)&(books['language'].isna())].index, 'language'] = candidate_lang
  except:
    pass


# # null 값 etc로 채우기
# books['language'].fillna('etc', inplace = True)


books['book_title']= books['book_title'].apply(lambda x: ''.join(x.split()).strip())
books['book_title']


# books의 카테고리 부분.
# 대괄호 써있는 카테고리 전처리
books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
# 모두 소문자로 통일
books['category'] = books['category'].str.lower()

# 미리 지정해놓은 category에 맞춰 category_high 칼럼 추가

categories_50 = [
        'Fiction', 'Science Fiction', 'Fantasy', 'Science', 'History', 'Autobiography', 'Mystery', 'Thriller', 'Romance', 'Politics',
        'Economics', 'Psychology', 'Philosophy', 'Religion', 'Sociology', 'Culture', 'Art', 'Music', 'Engineering', 'Computer Science',
        'Mathematics', 'Biology', 'Physics', 'Chemistry', 'Medicine', 'Family and Relationships', 'Cooking', 'Travel', 'Sports', 'Health and Wellness',
        'Nature', 'Environment', 'Animals', 'Plants', 'Technology', 'Business', 'Self-Help', 'Personal Growth', "Children's Books", 'Comics',
        'Young Adult', "Children's Picture Books", 'Science Books', 'Art and Photography', 'Poetry', 'Drama', 'Literary Fiction', 'Architecture',
        ]

books['category_high'] = books['category'].copy()
for category in categories_50:
    books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
    
    
    # 추가로 항목이 5개 이하인 것은 others로 분류
category_counts = books['category_high'].value_counts()
categories_to_others = category_counts[category_counts <= 10].index
books.loc[books['category_high'].isin(categories_to_others), 'category_high'] = 'others'


# 대부분의 country는 미국이기 때문에 결측치를 영어로 처리, category_high의 결측치는 fiction으로 처리
books['language'].fillna('en', inplace = True)
books['category_high'].fillna('fiction', inplace=True)
books['book_author'].fillna('other', inplace=True)


# 출판연도 범주화
# catboost 모델의 성능을 더 올리기 위해 데이터를 categorize

books['years'] = books['year_of_publication'].copy()
books['years'][books['year_of_publication'] < 1970] = 1970
books['years'][(books['year_of_publication'] < 1980) * (books['year_of_publication'] >= 1970)] = 1980
books['years'][(books['year_of_publication'] < 1990) * (books['year_of_publication'] >= 1980)] = 1990
books['years'][(books['year_of_publication'] < 2000) * (books['year_of_publication'] >= 1990)] = 2000
books['years'][(books['year_of_publication'] >= 2000)] = 2020
books['years'] = books['years'].astype('str')
#books['years'] = books['years'].astype('int')
books.drop(['year_of_publication', 'category'], axis = 1, inplace = True)







# 그대로 진행하는 것이 더 좋은 결과를 예측할 것.
users['age'].fillna(np.mean(users['age']), inplace=True)
def age_map(x):
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6
    
users['age'] = users['age'].apply(age_map)


users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거

users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.


# ffm_preprocessing 참고
# 원래는 지정해둔 cities로 하려고 했지만, 여기에서 처리하기에는 시간이 오래걸리는 것 같아 일단 default로 대체
modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]



# cities = [
#     'newyorkcity', 'losangeles', 'chicago', 'houston', 'phoenix',
#     'philadelphia', 'sanantonio', 'sandiego', 'dallas', 'sanjose',
#     'austin', 'jacksonville', 'sanfrancisco', 'columbus', 'fortworth',
#     'indianapolis', 'charlotte', 'seattle', 'denver', 'washington,d.c.',
#     'boston', 'elpaso', 'nashville', 'detroit', 'oklahomacity',
#     'portland', 'lasvegas', 'memphis', 'louisville', 'baltimore',
#     'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
#     'kansascity', 'longbeach', 'mesa', 'atlanta', 'coloradosprings',
#     'virginiabeach', 'raleigh', 'omaha', 'miami', 'oakland', 'minneapolis',
#     'tulsa', 'wichita', 'neworleans'
# ]



# 이렇게 하면 내부의 데이터를 사용하여 처리.
location_list = []
for location in modify_location:
    try:
        right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
        location_list.append(right_location)
    except:
        pass

for location in location_list:
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
    users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]

# city, state, country 딕셔너리 생성
loc_city2idx = {v:k for k,v in enumerate(users['location_city'].unique())}
loc_state2idx = {v:k for k,v in enumerate(users['location_state'].unique())}
loc_country2idx = {v:k for k,v in enumerate(users['location_country'].unique())}

# 딕셔너리를 기준으로 매핑 진행
users['location_city'] = users['location_city'].map(loc_city2idx)
users['location_state'] = users['location_state'].map(loc_state2idx)
users['location_country'] = users['location_country'].map(loc_country2idx)



users = users[['user_id', 'location_city', 'location_state', 'location_country','age']]


# 전처리 완료한 books와 users 테이블을 이용해 rating 테이블과 merge 하기.
# baseline참고
train_ratings = pd.read_csv(path+'train_ratings.csv')
test_ratings = pd.read_csv(path+'test_ratings.csv')

train_ratings = pd.merge(train_ratings,books, how='right',on='isbn')
train_ratings.dropna(subset=['rating'], inplace = True)
train_ratings = pd.merge(train_ratings, users, how='right',on='user_id')
train_ratings.dropna(subset=['rating'], inplace = True)

test_ratings['index'] = test_ratings.index
test_ratings = pd.merge(test_ratings,books, how='right',on='isbn')
test_ratings.dropna(subset=['rating'], inplace = True)
test_ratings = pd.merge(test_ratings, users, how='right',on='user_id')
test_ratings.dropna(subset=['rating'], inplace = True)
test_ratings = test_ratings.sort_values('index')
test_ratings.drop(['index'], axis=1, inplace=True)

train_ratings['user_id'] = train_ratings['user_id'].astype('str')
test_ratings['user_id'] = test_ratings['user_id'].astype('str')

train_ratings['location_city'] = train_ratings['location_city'].astype('str')
test_ratings['location_city'] = test_ratings['location_city'].astype('str')

train_ratings['location_state'] = train_ratings['location_state'].astype('str')
test_ratings['location_state'] = test_ratings['location_state'].astype('str')

train_ratings['location_country'] = train_ratings['location_country'].astype('str')
test_ratings['location_country'] = test_ratings['location_country'].astype('str')

train_ratings['rating'] = train_ratings['rating'].astype(int)
train_ratings['years'] = train_ratings['years'].astype(int)
train_ratings['age'] = train_ratings['age'].astype(int)

test_ratings['rating'] = test_ratings['rating'].astype(int)
test_ratings['years'] = test_ratings['years'].astype(int)
test_ratings['age'] = test_ratings['age'].astype(int)


train_ratings.drop(['mask_isbn','lang_isbn'],axis=1, inplace=True)


train_ratings = train_ratings.dropna()


from sklearn.model_selection import StratifiedKFold
fold_num = 5
skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
folds = []
for train_idx, valid_idx in skf.split(train_ratings, train_ratings['rating']):
    folds.append((train_idx,valid_idx))
    
    


from sklearn.model_selection import ParameterGrid


# 하이퍼파라미터 그리드
param_grid = {
    "learning_rate": [0.03],
    "max_depth": [8],
    "random_strength": [40],
    "l2_leaf_reg": [2.355742708217648e-05],
    "min_child_samples": [30],
    "max_bin": [300],
    "n_estimators": [3000]
}

# 고정 파라미터
fixed_params = {
    "random_state": 42,
    "objective": "RMSE",
    # "od_type" : 'Iter' ,
    "bagging_temperature" : 0.1,
    "task_type": "GPU",
    "devices": "0",
    "cat_features": list(train_ratings.drop(['rating'], axis=1).columns)
}

# 그리드 서치 수행
best_rmse = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    rmse_scores = []

    for fold in range(fold_num):
        train_idx, valid_idx = folds[fold]
        X_train = train_ratings.drop(['rating'], axis=1).iloc[train_idx]
        X_valid = train_ratings.drop(['rating'], axis=1).iloc[valid_idx]
        y_train = train_ratings['rating'].iloc[train_idx]
        y_valid = train_ratings['rating'].iloc[valid_idx]

        # 모델 생성 및 학습
        model = CatBoostRegressor(**params, **fixed_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)

        # 검증 데이터에 대한 예측 및 RMSE 계산
        valid_pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        rmse_scores.append(rmse)

    mean_rmse = np.mean(rmse_scores)
    print(f"Params: {params}, Mean RMSE: {mean_rmse}")

    # 최적의 파라미터 업데이트
    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_params = params

# 최적의 파라미터 출력
print(f"Best RMSE: {best_rmse}")
print(f"Best Parameters: {best_params}")


test_ratings['rating'] = (test_ratings['pred_0'] + test_ratings['pred_1'] + test_ratings['pred_2'] + test_ratings['pred_3'] + test_ratings['pred_4']) / fold_num
# 
submit = test_ratings[['user_id', 'isbn', 'rating']]
submit.to_csv('../submit/CAT_5Fold.csv', index = False)