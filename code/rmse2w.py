import argparse
import numpy as np

def main(args):
    config = {
        "esb_opt" : args.esb_opt,
        "b_rmse" : args.b_rmse,
        "d_rmse" : args.d_rmse,
        "rmse" : args.rmse,
    }
    b_rmse=np.array(args.b_rmse.split(","), dtype=float)
    d_rmse=np.array(args.d_rmse.split(","), dtype=float)
    rmse=np.array(args.rmse.split(","), dtype=float)

    if args.esb_opt == 'div':
        b_rmse=1/b_rmse
        d_rmse=1/d_rmse
        
        b_w=b_rmse.mean()/(b_rmse.mean()+d_rmse.mean())
        d_w=d_rmse.mean()/(b_rmse.mean()+d_rmse.mean())
        b_rmse_ratio=b_rmse/b_rmse.sum()
        d_rmse_ratio=d_rmse/d_rmse.sum()
        

        w=b_w*b_rmse_ratio
        w=np.concatenate((w,d_w*d_rmse_ratio))
        print(w)
        
    elif args.esb_opt == 'basic':
        rmse=1/rmse
        
        w=rmse/rmse.sum()
        print(w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--esb_opt', type=str, choices=['div','basic']
        , help='앙상블 전략을 선택, [div, basic] div:boost모델과 DL모델로 나누어 계산, basic: 모델상관없이 rmse기준 weight계산')
    arg('--b_rmse', type=str, default='0', help='boost 모델의 rmse를 순서대로 입력, ex. 2.1,3.0,2.2,2.3')
    arg('--d_rmse', type=str, default='0', help='DL 모델의 rmse를 순서대로 입력, ex. 2.1,3.0,2.2,2.3')
    arg('--rmse', type=str, default='0', help='모델의 rmse를 순서대로 입력, ex. 2.1,3.0,2.2,2.3')

    args = parser.parse_args()
    main(args)