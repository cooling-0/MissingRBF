import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models import train_BiMGRU, BiMGRU_RBF
from utils import missing_data_rbf, eval_model2_ver2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--lr', type = float, default=1e-3)

parser.add_argument('--input_size', type = int, default = 7)
parser.add_argument('--hidden_size', type= int, default = 64)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--seqlen', type = int, default = 24)
parser.add_argument('--missing', type = str, default = 'long')
parser.add_argument('--models', type = str, default = 'RBF')

args = parser.parse_args()

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = './result/'
    model_name = 'G_{}_{}_{}_{}_year'.format(args.lr, args.hidden_size, args.seqlen, args.models)
    print('device:',device)
    if args.missing == 'long':
        df = pd.read_csv('./dataset/ETh_year_longterm.csv').drop(['date'], axis = 1)
        print('long missing train')
        rbf = "rbf_year_{}_20.csv".format(args.missing)
        dataset = missing_data_rbf(df, rbf, args.batch_size, args.seqlen)
        G = BiMGRU_RBF(args.input_size, args.hidden_size)
        loss_list = train_BiMGRU(G, args.lr, args.epochs, dataset, device)
        torch.save(G, save_path + model_name + '.pt')
        Nonscale_imputataion = eval_model2_ver2(G, rbf, df)


    else:
        print("short missing train")
        df = pd.read_csv('./dataset/df_year20missing.csv').drop(['date'], axis = 1)
        rbf = "rbf_year_{}_20.csv".format(args.missing)
        dataset = missing_data_rbf(df, rbf, args.batch_size, args.seqlen)

        G = BiMGRU_RBF(args.input_size, args.hidden_size)
        loss_list = train_BiMGRU(G, args.lr, args.epochs, dataset, device)
        torch.save(G, save_path + model_name + '.pt')
        Nonscale_imputataion = eval_model2_ver2(G, rbf, df)
    
    Nonscale_imputataion.to_csv('./result/{}_MIRNN-CF_year.csv'.format(args.missing),index = False)

if __name__ == '__main__':
    run()