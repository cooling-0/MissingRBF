import pandas as pd
import numpy as np
import torch

from time import time
from utils import Dataset, make_rbf_df
from models import MultiRBFnnTime, MultiRBFnn_sigma

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 3000)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--add_rbf_num', type=int, default=50)
parser.add_argument('--date', type= str, default='half')
parser.add_argument('--missing', type= str, default='long')
parser.add_argument('--sigma', type= str, default='time')

args = parser.parse_args()


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print()

    if args.sigma == "time":
        print("{} data {} missing train".format(args.date, args.missing))
        save_name = "rbf_{}_{}_{}.csv".format(args.date, args.missing, args.add_rbf_num)
        save_path = "./RBFresult/"
        save_model = "{}_rbf_{}_{}.pt".format(args.date, args.missing, args.add_rbf_num)
        model_save_path = "./RBFresult/"

        input_data, target, lossth, target_ground, missing_index = Dataset(args.date, args.missing, device)
        print("loss th:", lossth)
        model = MultiRBFnnTime(target.size(0), add_rbf_num = args.add_rbf_num, device=device)
        model.train(input_data = input_data, target = target, epochs = 3000,
                        lr = args.lr, loss_th=lossth, lr_change_th=lossth)
        
        model_pred = model.pred(input_data)[1]

        print(torch.mean(torch.abs(target_ground[(missing_index != 1)] -  model_pred[(missing_index != 1)])))
        print(torch.mean(torch.abs(target_ground[(missing_index != 0)] -  model_pred[(missing_index != 0)])))

        rbf_df = make_rbf_df(model_pred)

        torch.save(model, model_save_path + save_model)
        rbf_df.to_csv(save_path + save_name, index = False)
    else:
        print("{} data {} missing train".format(args.date, args.missing))
        save_name = "rbf_{}_{}_{}.csv".format(args.date, args.missing, args.add_rbf_num)
        save_path = "./RBFresultsigma/"
        save_model = "{}_rbf_{}_{}.pt".format(args.date, args.missing, args.add_rbf_num)
        model_save_path = "./RBFresultsigma/"

        input_data, target, lossth, target_ground, missing_index = Dataset(args.date, args.missing, device)
        print("loss th:", lossth)
        model = MultiRBFnn_sigma(target.size(0), add_rbf_num = args.add_rbf_num, device=device)
        model.train(input_data = input_data, target = target, epochs = 3000,
                        lr = args.lr, loss_th=lossth, lr_change_th=lossth)
        
        model_pred = model.pred(input_data)[1]

        print(torch.mean(torch.abs(target_ground[(missing_index != 1)] -  model_pred[(missing_index != 1)])))
        print(torch.mean(torch.abs(target_ground[(missing_index != 0)] -  model_pred[(missing_index != 0)])))

        rbf_df = make_rbf_df(model_pred)

        torch.save(model, model_save_path + save_model)
        rbf_df.to_csv(save_path + save_name, index = False)

if __name__ == '__main__':
    run()
    






