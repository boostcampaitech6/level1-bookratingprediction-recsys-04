import isbnlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# def ffm_age_map(x : int) -> int :
#     if pd.isnull(x) :
#         return 9
#     if x ==np.nan :
#         return 9
    
#     if x <10 :
#         return 0
#     elif 10 <= x < 20 :
#         return 1
#     elif 20 <= x < 30 :
#         return 2
#     elif 30 <= x < 40 :
#         return 3
#     elif 40 <= x < 50 :
#         return 4
#     elif 50 <= x < 60 :
#         return 5
#     elif 60 <= x < 70 :
#         return 6
#     elif 70 <= x < 80 :
#         return 7
#     elif 80 <= x < 90 :
#         return 8
#     else :
#         return 9



def modifying_location(users) :
    '''
    도시의 city가 있는 country가 결측인 데이터를 찾아 처리해주는 함수입니다.
    '''
    # 결측치를 가진 도시 위치 데이터 찾기
    cities = [
    'newyorkcity', 'losangeles', 'chicago', 'houston', 'phoenix',
    'philadelphia', 'sanantonio', 'sandiego', 'dallas', 'sanjose',
    'austin', 'jacksonville', 'sanfrancisco', 'columbus', 'fortworth',
    'indianapolis', 'charlotte', 'seattle', 'denver', 'washington,d.c.',
    'boston', 'elpaso', 'nashville', 'detroit', 'oklahomacity',
    'portland', 'lasvegas', 'memphis', 'louisville', 'baltimore',
    'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
    'kansascity', 'longbeach', 'mesa', 'atlanta', 'coloradosprings',
    'virginiabeach', 'raleigh', 'omaha', 'miami', 'oakland', 'minneapolis',
    'tulsa', 'wichita', 'neworleans'
]

    modify_location = users[(users['location_country'].isna()) & (users['location_city'].notnull())]['location_city'].values

    location_list = []
    for city in cities:
        for location in modify_location:
            try:
                right_location = users[(users['location'].str.contains(location)) & (users['location_country'].notnull())]['location'].value_counts().index[0]
                location_list.append(right_location)
            except:
                pass

    for location in location_list:
        city, state, country = location.split(',')
        users.loc[users[users['location_city'] == city].index, 'location_state'] = state
        users.loc[users[users['location_city'] == city].index, 'location_country'] = country


    return users


# def fill_pub_info(books) :
#     '''
#     isbn lib를 활용하여 출판사의 정보를 불러와 결측값을 채웁니다.
#     사용할 수 없으면 가장 대중적인 언어로 처리하는 방식을 사용합니다.
#     '''
#     def getlang(isbn) :
#         try :
#             book_info = isbnlib.meta(isbn)
#             # print(book_info)
#             language = book_info.get('Language', np.nan)
#         except :
#             # print('error')
#             language = np.nan
#         return language

#     books['language'] = books['isbn'].apply(getlang)

#     return books


def categorize_and_encode(x):
    '''
    지정해놓은 임의의 category 분류리스트를 바탕으로 해당하는 카테고리가 있는
    카테고리 칼럼에 대해 category_high 칼럼에 해당하는 카테고리 값을 넣습니다.
    최종적으로 없는 부분은 "others"로 처리합니다.
    
    그 다음 이를 label encoding하여 field 값으로 변환시키고, category_high를 제거합니다.
    '''
    # 먼저 결측치를 others로 채웁니다.
    x['category'].fillna('others', inplace=True)

    # 나머지 list 문자열을 일반 문자열로 반환합니다.
    def lst_to_str(x) :
        if x != 'others' :
            return x[2:-2]
        else :
            return x
    x['category'] = x['category'].apply(lst_to_str)


    # 새로운 category_high를 만듭니다.
    x['category_high'] = x['category'].str.lower()

    # 카테고리를 지정한 후, 소문자로 반환합니다.
    categories_50 = [
        'Fiction', 'Science Fiction', 'Fantasy', 'Science', 'History', 'Autobiography', 'Mystery', 'Thriller', 'Romance', 'Politics',
        'Economics', 'Psychology', 'Philosophy', 'Religion', 'Sociology', 'Culture', 'Art', 'Music', 'Engineering', 'Computer Science',
        'Mathematics', 'Biology', 'Physics', 'Chemistry', 'Medicine', 'Family and Relationships', 'Cooking', 'Travel', 'Sports', 'Health and Wellness',
        'Nature', 'Environment', 'Animals', 'Plants', 'Technology', 'Business', 'Self-Help', 'Personal Growth', "Children's Books", 'Comics',
        'Young Adult', "Children's Picture Books", 'Science Books', 'Art and Photography', 'Poetry', 'Drama', 'Literary Fiction', 'Architecture',
        ]

    lowercase_categories = [category.lower() for category in categories_50]

    # 카테고리에 해당하는 부분을 찾고 없으면 others로 반환합니다.
    x.loc[~x['category_high'].isin(lowercase_categories), 'category_high'] = 'others'

    # 추가로 10개 이하인 항목은 'others'로 묶어주도록 합니다.
    category_counts = x['category_high'].value_counts()
    categories_to_others = category_counts[category_counts <= 100].index
    x.loc[x['category_high'].isin(categories_to_others), 'category_high'] = 'others'

    # 마지막으로 나온 결과를 LabelEncoding합니다.
    from sklearn.preprocessing import LabelEncoder
    # tqdm.pandas()
    label_encoder = LabelEncoder()
    x['category_high_encode'] = label_encoder.fit_transform(x['category_high'])

    return x
