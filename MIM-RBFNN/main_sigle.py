import pandas as pd
import numpy as np
import torch

from time import time
from utils_single import Dataset, make_rbf_df
from models import MultiRBFnnTime, MultiRBFnn_sigma

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 3000)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--add_rbf_num', type=int, default=50)
parser.add_argument('--date', type= str, default='year')
parser.add_argument('--missing', type= str, default='long')
parser.add_argument('--sigma', type= str, default='time')

args = parser.parse_args()

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()

    if args.sigma == "time":
        print("{} data {} missing {} train".format(args.date, args.missing, args.sigma))
        save_name = "rbf_{}_{}_{}.csv".format(args.date, args.missing, args.add_rbf_num)
        save_path = "./RBFresult/"
        save_model = "{}_rbf_{}_{}.pt".format(args.date, args.missing, args.add_rbf_num)
        model_save_path = "./RBFresult/"
        pred_list = []
        input_data, target, lossth, target_ground, missing_index = Dataset(args.date, args.missing, device)
        print("loss th:", lossth)
        for i in range(target.size(0)):
            print("{}th train start".format(i))
            model = MultiRBFnnTime(1, add_rbf_num = args.add_rbf_num, device=device)
            model.train(input_data = input_data, target = target[i], epochs = 3000,
                            lr = args.lr, loss_th=lossth[i], lr_change_th=lossth[i])
        
            model_pred = model.pred(input_data)[1]
            torch.save(model, model_save_path + save_model + "{}".format(i))
            print("{}th loss".format(i))
            print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 1)] -  model_pred[(missing_index[i] != 1)])))
            print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 0)] -  model_pred[(missing_index[i] != 0)])))
            pred_list.append(model_pred)
        rbf_df = make_rbf_df(pred_list)

        rbf_df.to_csv(save_path + '_single' + save_name , index = False)
    else:
        print("{} data {} missing {} train".format(args.date, args.missing, args.sigma))
        save_name = "rbf_{}_{}_{}.csv".format(args.date, args.missing, args.add_rbf_num)
        save_path = "./RBFresultsigma/"
        save_model = "{}_rbf_{}_{}.pt".format(args.date, args.missing, args.add_rbf_num)
        model_save_path = "./RBFresultsigma/"

        pred_list = []
        input_data, target, lossth, target_ground, missing_index = Dataset(args.date, args.missing, device)
        print("loss th:", lossth)
        for i in range(target.size(0)):
            print("{}th train start".format(i))
            model = MultiRBFnn_sigma(1, add_rbf_num = args.add_rbf_num, device=device)
            model.train(input_data = input_data, target = target[i], epochs = 3000,
                            lr = args.lr, loss_th=lossth[i], lr_change_th=lossth[i])
            
            model_pred = model.pred(input_data)[1]
            torch.save(model, model_save_path + save_model + "{}".format(i))

            print("{}th loss".format(i))
            print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 1)] -  model_pred[(missing_index[i] != 1)])))
            print(torch.mean(torch.abs(target_ground[i][(missing_index[i] != 0)] -  model_pred[(missing_index[i] != 0)])))
            pred_list.append(model_pred)

        rbf_df = make_rbf_df(pred_list)
        rbf_df.to_csv(save_path + '_single' + save_name, index = False)

if __name__ == '__main__':
    run()