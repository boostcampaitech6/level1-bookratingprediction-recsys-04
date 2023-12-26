import argparse
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import re
from tqdm import tqdm
import isbnlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor


def main(args):
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    # books['language'] 결측치 처리

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
            candidate_lang = books.loc[books['lang_isbn']
                                       == i]['language'].value_counts().index[0]
            books.loc[books[(books['lang_isbn'] == i) & (
                books['language'].isna())].index, 'language'] = candidate_lang
        except:
            pass

    # null 값 etc로 채우기
    books['language'].fillna('etc', inplace=True)

    # book['category'] 결측치 처리 및 isbnlib 분할

    # []제거, 소문자처리
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull(
    )]['category'].apply(lambda x: re.sub('[\W_]+', ' ', x).strip())  # category [''] 제거
    books['category'] = books['category'].str.lower()

    # category 결측치 <- 작가
    # 작가별 가장 많이 나오는 카테고리 계산 (NA 무시)
    author_top_category = books.dropna(subset=['category']).groupby('book_author')['category'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    # 카테고리가 NA인 책의 카테고리를 해당 작가의 가장 많이 나오는 카테고리로 채우기
    books['category'] = books.apply(lambda row: author_top_category.get(
        row['book_author'], np.nan) if pd.isnull(row['category']) else row['category'], axis=1)
    # 결과:결측치 27203개 남음

    # category 결측치 <- 출판사
    # 출판사별 가장 많이 나오는 카테고리 계산 (NA 무시)
    publisher_top_category = books.dropna(subset=['category']).groupby('publisher')['category'].agg(
        lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    # 카테고리가 NA인 책의 카테고리를 해당 출판사의 가장 많이 나오는 카테고리로 채우기
    books['category'] = books.apply(lambda row: publisher_top_category.get(
        row['publisher'], np.nan) if pd.isnull(row['category']) else row['category'], axis=1)
    # 결과:결측치 4691개 남음

    # 카테고리의 NaN <- etc
    books['category'] = books['category'].fillna('etc')
    # 결과:결측치 0개 남음

    books['isbn'] = books['isbn'].astype(str)
    isbn_df = books['isbn'].apply(
        lambda x: isbnlib.mask(x) if isbnlib.is_isbn10(x) else x)
    books['Country identifier'] = isbn_df.apply(
        lambda x: x.split('-')[0] if '-' in x else None)
    books['Publisher identifier'] = isbn_df.apply(lambda x: x.split(
        '-')[1] if '-' in x and len(x.split('-')) > 1 else None)

    # 기존 코드
    # train, test의 user_id, isbn 통합
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    # user_id, isbn 인덱싱
    idx2user = {idx: id for idx, id in enumerate(ids)}
    idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}
    user2idx = {id: idx for idx, id in idx2user.items()}
    isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

    # 인덱싱 적용
    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)
    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    # location 분리
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(
        lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(
        lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    # ratings, users, books 통합
    ratings = pd.concat([train, test]).reset_index(drop=True)
    context_df = ratings.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = train.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = test.merge(users, on='user_id', how='left').merge(
        books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # location 인덱싱
    loc_city2idx = {v: k for k, v in enumerate(
        context_df['location_city'].unique())}
    loc_state2idx = {v: k for k, v in enumerate(
        context_df['location_state'].unique())}
    loc_country2idx = {v: k for k, v in enumerate(
        context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(
        loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(
        loc_country2idx)

    # age 결측치 처리 / 범주화

    def age_map(x: int) -> int:
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

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # books feature 인덱싱
    category2idx = {v: k for k, v in enumerate(
        context_df['category'].unique())}
    publisher2idx = {v: k for k, v in enumerate(
        context_df['publisher'].unique())}
    language2idx = {v: k for k, v in enumerate(
        context_df['language'].unique())}
    author2idx = {v: k for k, v in enumerate(
        context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop(
        ['rating'], axis=1), train_df['rating'], test_size=0.2, shuffle=True)

    model = CatBoostRegressor(task_type='GPU', devices='0')

    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    print('RMSE: ', rmse)

    # H2O AutoML

    X_train_copy = X_train.copy()
    X_train_copy['rating'] = y_train
    training_frame = h2o.H2OFrame(X_train_copy)

    x_h2o = training_frame.columns
    y_h2o = 'rating'
    x_h2o.remove(y_h2o)

    h2o.init()

    aml = H2OAutoML(
        max_models=10,
        seed=42,
        max_runtime_secs=360,
        sort_metric='RMSE'

    )

    # h2o.connect()

    aml.train(
        x=x_h2o,
        y=y_h2o,
        training_frame=training_frame
    )

    leaderboard = aml.leaderboard
    # print(leaderboard.head())

    X_valid_copy = X_valid.copy()
    valid_frame = h2o.H2OFrame(X_valid_copy)

    # print(len(valid_frame))

    model_h2o = aml.leader

    X_test_copy = test_df.copy()
    X_test_copy.drop(columns=['rating'])

    # valid
    test_pred_h2o = model_h2o.predict(X_test_copy)

    test_pred_df_h2o = pd.DataFrame(test_pred_h2o)

    # sub = pd.read_csv(path + 'sample_submission.csv')

    sub['rating'] = test_pred_df_h2o.values

    filename = f'./submit/H2OAutoML_pred.csv'
    sub.to_csv(filename, index=False)


if __name__ == "__main__":

    # BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    # BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')

    args = parser.parse_args()
    main(args)
