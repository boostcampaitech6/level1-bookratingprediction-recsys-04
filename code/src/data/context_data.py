import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

# feat siyun : FFM을 위한 전처리 함수를 import 합니다.
# 추가적으로 isbnlib를 사용한다면 이에 맞는 라이브러리를 불러옵니다.
# 라벨 인코딩을 진행하기 때문에 라이브러리를 불러옵니다.
from .ffm_preprocessing import * 
import isbnlib
from sklearn.preprocessing import LabelEncoder



# feat siyun - age의 category 범위를 9까지 증가
# 그 외 남아있을 null, nan값은 9로 치환합니다.
def age_map(x: int) -> int:
    if pd.isnull(x) :
        return 9
    if x ==np.nan :
        return 9
    x = int(x)
    
    if x <10 :
        return 0
    elif 10 <= x < 20 :
        return 1
    elif 20 <= x < 30 :
        return 2
    elif 30 <= x < 40 :
        return 3
    elif 40 <= x < 50 :
        return 4
    elif 50 <= x < 60 :
        return 5
    elif 60 <= x < 70 :
        return 6
    elif 70 <= x < 80 :
        return 7
    elif 80 <= x < 90 :
        return 8
    else :
        return 9


# users, books, train, test
def process_context_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    feat siyun - modifying_location, categorize_and_encode 추가
    - modifying_location(x) : 지역의 city를 찾아 country가 결측인 부분을 채우는 함수
    - categorize_and_encode(x) : 지정해 놓은 n개의 category를 바탕으로 해당하는 부분을 category_high 칼럼으로 새롭게 받아 Label_encoding
    args를 받아 진행하고 싶었지만, 본 함수에서는 arg를 받지 않기 때문에 다음과 같이 강제로 추가하는 방식을 일단 진행하였습니다. 추후 변경 예정.
    """


    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    # feat siyun - city와 관련된 내용 처리할 예정
    # train,test는 users에서 분할하기 때문에 분할하기 전 전처리를 진행합니다.
    # books에 대해 v2idx를 진행하기 전 새로운 category_high카테고리를 만든 후 category_high_encode를 추가적으로 생성합니다
    #--------------------------
    users = modifying_location(users)
    books = categorize_and_encode(books)
    #-----------------------
    
    
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    # 레이팅 처리
    ## feat siyun - category_high 칼럼 추가
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high_encode']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high_encode']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'category_high_encode']], on='isbn', how='left')

    
    
    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    # feat siyun - age 처리
    # 평균으로 치환했지만, 이미 처리를 했기 떄문에 해당 부분은 지우겠습니다.
    # train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    # test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)



    # book 파트 인덱싱
    
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    
    # feat siyun - 새롭게 추가합니다.
    # v,k 그대로 적용합니다.
    category_high_encode = {v:k for v,k in enumerate(context_df['category_high_encode'].unique())}


    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    # feat siyun - idx에 새로 추가한 칼럼 category_high_encode를 추가했습니다.
    # idx['category_~~]를 넣으면 그 값으로 LabelEncoding이 반환됩니다.
    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        "category_high_encode" : category_high_encode,
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """
    
    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

# # feat siyun 
# # args.model이 FFM이면 field_dims를 새로만든 columns을 필드로 추가합니다.
# 여러 변수를 고려해서 큰 의미가 없다고 판단되는 변수는 제거하겠습니다.
    if hasattr(args, 'model') and args.model == 'FFM':
        field_dims = np.array([len(user2idx), len(isbn2idx),
                           10, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                           len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), 
                           len(idx['author2idx']),len(idx['category_high_encode'])
                           ], dtype=np.uint32)


    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
