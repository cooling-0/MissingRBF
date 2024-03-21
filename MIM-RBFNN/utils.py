import pandas as pd
import numpy as np
import torch

def Dataset(date, missing, device):
    
    if missing == 'long':
        df = pd.read_csv('./dataset/ETh_{}_longterm.csv'.format(date)).drop(['date'], axis = 1)
    else:
        df = pd.read_csv('./dataset/df_{}20missing.csv'.format(date)).drop(['date'], axis = 1)
    df_ground = pd.read_csv('./dataset/ETh_{}.csv'.format(date)).drop(['date'], axis = 1)
    lossth = np.mean(df.mean().values) * 0.06 

    input_data =  torch.tensor(np.array(df.index) + 1, device = device, dtype = torch.float64)
    target = torch.tensor(df.values.T.astype(np.float64), device = device, dtype = torch.float64)
    target_ground = torch.tensor(df_ground.values.T.astype(np.float64), device = device, dtype = torch.float64)

    train_masking = torch.where(target.isnan(), 0.0 , 1.0)
    real_masking = torch.where(target_ground.isnan(), 0.0 , 1.0)
    missing_index = real_masking - train_masking
    print(target.size())
    print(torch.sum(missing_index))
    

    return input_data, target, lossth, target_ground, missing_index


def make_rbf_df(model_pred):


    rbf_df = pd.DataFrame(model_pred.T.detach().cpu().numpy(), columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])

    return rbf_df
