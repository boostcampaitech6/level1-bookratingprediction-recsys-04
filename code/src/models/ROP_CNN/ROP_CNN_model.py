import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from sklearn.decomposition import TruncatedSVD
from scipy.signal import convolve2d

# Rating Prediction
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.clamp(x, min=0, max=10)
        return x

# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):   #user id embedding
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

# CNN in Review Text Encoder
class CNN_1D(nn.Module):
    def __init__(self, word_dim, out_dim, kernel_size, conv_1d_out_dim):
        super(CNN_1D, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv1d(  ##Convolution
                                        in_channels=5,#word_dim,  #입력 데이터의 차원 수
                                        out_channels=out_dim,  #출력 데이터의 차원 수
                                        kernel_size=kernel_size,  #커널의 크기
                                        padding=(kernel_size - 1) // 2),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=kernel_size), ##Max-pooling
                                nn.Dropout(p=0.5)
                                    )
        self.bn = nn.BatchNorm1d(out_dim)
        self.linear = nn.Sequential(  ##Fully Connected
                                    nn.Linear(out_dim * (word_dim // kernel_size), conv_1d_out_dim), #Multiple 지점
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3)
                                    )
    def forward(self, vec):
        output = self.conv(vec)
        output = self.bn(output)
        output = output.reshape(output.size(0), -1) 
        output = self.linear(output)
        return output

# CNN in Interaction Encoder
class CNN_2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_2d_out_dim):
        super(CNN_2D, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size - 1) // 2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=kernel_size),
                                nn.Dropout(p=0.5)
                                    )
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Sequential(
                                    nn.Linear(out_channels * (100 // kernel_size) * (100 // kernel_size), conv_2d_out_dim), #Multiple 지점
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3)
                                    )
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1) 
        x = self.linear(x)
        return x



    
# Review text encoder와 interaction encoder를 결합하여 Denselayer로 학습
class ROP_CNN(nn.Module):
    def __init__(self, args, data):
        super(ROP_CNN, self).__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.deepconn_embed_dim)
        
        ## U_I Rating Matrix -> U & I Latent Factor Matrix

        # TruncatedSVD 모델 생성
        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        # 사용자-아이템 평점 행렬에 대해 SVD 적용
        user_factors = svd.fit_transform(data['users_items_rating_matrix'])
        # 사용자 잠재 요인 행렬
        self.user_factors = nn.Parameter(torch.from_numpy(user_factors).float())
        # 아이템 잠재 요인 행렬
        self.item_factors = nn.Parameter(torch.from_numpy(svd.components_.T).float())

        self.cnn_u = CNN_2D( #interaction encoder의 2D CNN
                             in_channels=5,
                             out_channels=args.out_dim,
                             kernel_size=args.kernel_size,
                             conv_2d_out_dim=args.conv_1d_out_dim,
                            )
        self.cnn_i = CNN_1D( #Review text encoder의 1D CNN
                             word_dim=args.word_dim,
                             out_dim=args.out_dim,
                             kernel_size=args.kernel_size,
                             conv_1d_out_dim=args.conv_1d_out_dim,
                            )
        self.MLP = MLP( #Rating Prediction의 MLP
                            input_dim=args.conv_1d_out_dim*2, 
                            hidden_dim=args.conv_1d_out_dim,
                            output_dim=1
                            )
    def forward(self, x):
        user_isbn_vector, item_text_vector = x[0], x[2]

        #user_isbn_feature = self.embedding(user_isbn_vector) #사용자의 ISBN를 저차원 벡터로 임베딩
        
        ## Review Text Encoder
        item_text_vector = item_text_vector.permute(0,2,1)  # item_text_vector : Batch수(1024)*embedding 수(768)*1
        item_text_vector = item_text_vector.repeat(1, 5, 1)
        item_text_feature = self.cnn_i(item_text_vector) 
        
        ## Interaction Encoder
        # 배치 내의 사용자와 아이템 인덱스 추출
        user_indices = user_isbn_vector[:, 0].to(self.user_factors.device)
        item_indices = user_isbn_vector[:, 1].to(self.item_factors.device)
        # 필요한 사용자와 아이템의 잠재 요인 선택
        user_factor_batch = self.user_factors[user_indices]
        item_factor_batch = self.item_factors[item_indices]
        # 상호작용 맵 계산
        interaction_map = torch.bmm(user_factor_batch.unsqueeze(2), item_factor_batch.unsqueeze(1))
        # 상호작용 맵의 차원을 [batch_size, height, width]에서 [batch_size, 3, height, width]로 변경
        interaction_map = interaction_map.unsqueeze(1)  # 먼저 채널 차원을 추가합니다.
        interaction_map = interaction_map.repeat(1, 5, 1, 1)  # 채널 차원을 3으로 확장합니다.
        # 상호작용 맵을 3D CNN에 입력
        interaction_feature = self.cnn_u(interaction_map)
        
        ## Rating Prediction
        # Concatenate 
        feature_vector = torch.cat([interaction_feature,
                                    item_text_feature],
                                    dim=1)        
        # MLP
        output = self.MLP(feature_vector)
        return output.squeeze(1)
