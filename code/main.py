import time
import wandb
import argparse
import pandas as pd
from datetime import datetime
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test


def main(args):
    Setting.seed_everything(args.seed)
    global wandb_id
    wandb_id = wandb.util.generate_id()
    config = {

        "model": args.model,
        "data_shuffle": args.data_shuffle,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_fn": args.loss_fn,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "embed_dim": args.embed_dim,
        "dropout": args.dropout,
        "mlp_dims": args.mlp_dims,
        "num_layers": args.num_layers,
        "cnn_embed_dim": args.cnn_embed_dim,
        "cnn_latent_dim": args.cnn_latent_dim,
        "vector_create": args.vector_create,
        "deepconn_embed_dim": args.deepconn_embed_dim,
        "deepconn_latent_dim": args.deepconn_latent_dim,
        "conv_1d_out_dim": args.conv_1d_out_dim,
        "kernel_size": args.kernel_size,
        "word_dim": args.word_dim,
        "out_dim": args.out_dim,
        "clean_img_ver": args.clean_img_ver,
        "img_resize": args.img_resize,
        "img_transforms": args.img_transforms,
        "cnn_fm_ver": args.cnn_fm_ver,
    }

    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    wandb.init(id = wandb_id, resume = "allow", project= args.project, name = f'{args.model}_{timestamp}', config = config, entity=args.entity)

    # DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM'):
        data = context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.model == 'CNN_FM':
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)

    elif args.model == 'ROP_CNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)    
    else:
        pass

    # Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model == 'CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model == 'DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)

        
    elif args.model=='ROP_CNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    else:
        pass

    # Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()

    # Setting for wandb
    wandb.init(id=wandb_id, resume="allow", project=args.project,
               name=setting.get_wandb_name(args), config=config)

    # Model
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args, data)

    # TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model = train(args, model, data, logger, setting)

    # INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data, setting)

    # SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')

    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'ROP_CNN'):

        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":

    # BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    # WANDB OPTION
    arg('--project', type=str, default='book_project', help='프로젝트 이름을 설정할 수 있습니다.')
    arg('--entity', type=str, default='recsys4',
        help='Username 이나 Team name 을 설정할 수 있습니다.')

    # EDA OPTION
    arg('--clean_img_ver', type=int, default=0,
        choices=[0, 1], help='clean_books.csv 파일 사용 여부를 설정할 수 있습니다.')
    arg('--img_resize', type=int, default=32,
        help='이미지 전처리 시 resize 할 크기를 설정할 수 있습니다.')
    arg('--img_transforms', type=str, default='Resize',
        choices=['Resize', 'RandomResizedCrop'], help='이미지 transforms 방법을 설정할 수 있습니다.')
    arg('--save_eda_path', type=str, default='eda', help='EDA 저장 위치를 사용할 수 있습니다.')

    # BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models',
        help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'ROP_CNN'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2,
        help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True,
        help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')

    # TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE',
        choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM',
        choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6,
        help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    # GPU
    arg('--device', type=str, default='cuda',
        choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    # FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16,
        help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2,
        help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16),
        help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')

    # DCN
    arg('--num_layers', type=int, default=3,
        help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')

    # CNN_FM
    arg('--cnn_embed_dim', type=int, default=64,
        help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12,
        help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--cnn_fm_ver', type=int, default=1, help='CNN_FM 버전')

    # DeepCoNN & ROP_CNN
    arg('--vector_create', type=bool, default=False,
        help='DEEP_CONN,ROP_CNN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32,
        help='DEEP_CONN,ROP_CNN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10,
        help='DEEP_CONN,ROP_CNN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50,
        help='DEEP_CONN,ROP_CNN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3,
        help='DEEP_CONN,ROP_CNN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768,
        help='DEEP_CONN,ROP_CNN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32,
        help='DEEP_CONN,ROP_CNN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

    args = parser.parse_args()
    main(args)
