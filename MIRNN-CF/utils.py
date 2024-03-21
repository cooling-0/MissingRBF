from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_deltas(masks):
    deltas = []
    for h in range(len(masks)):
        if h == 0:
            deltas.append([1 for _ in range(masks.shape[1])])
        else:
            deltas.append([1 for _ in range(masks.shape[1])] + (1-masks[h]) * deltas[-1])
    
    return list(deltas)


class MyDataset(Dataset):
    def __init__(self, dataset, q):
        self.data = dataset
        self.q = q

    def __len__(self):
        return self.data.shape[1] // self.q

    def __getitem__(self, index):
        return self.data[:,index * self.q : index * self.q + self.q,:]

def missing_data_rbf(df, rbf, batch_size, seq_len):
    
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape
    rbf_df = pd.read_csv("./RBFresult/" + rbf)
    rbf_df_values = ((rbf_df - df.mean()) / rbf_df.std()).values
    masks = ~np.isnan(values)
    
    masks = masks.reshape(shp)

    deltas = np.array(make_deltas(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)
    rbf_x = torch.from_numpy(rbf_df_values).to(torch.float32)
    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0), rbf_x.unsqueeze_(0)], dim = 0)
    
    mydata  = MyDataset(dataset, seq_len)
    data = DataLoader(mydata, batch_size, shuffle=False)

    return data

def val_missing_data_rbf2(df, rbf):
    
    values = ((df - df.mean()) / df.std()).values
    shp = values.shape
    rbf_df = pd.read_csv("./RBFresult/" + rbf)
    rbf_df_values = ((rbf_df - df.mean()) / rbf_df.std()).values
    
    masks = ~np.isnan(values)
    
    masks = masks.reshape(shp)

    deltas = np.array(make_deltas(masks))
    values = torch.nan_to_num(torch.from_numpy(values).to(torch.float32))
    masks = torch.from_numpy(masks).to(torch.float32)
    deltas = torch.from_numpy(deltas).to(torch.float32)
    rbf_x = torch.from_numpy(rbf_df_values).to(torch.float32)
    dataset = torch.cat([values.unsqueeze_(0), deltas.unsqueeze_(0), masks.unsqueeze_(0), rbf_x.unsqueeze_(0)], dim = 0).unsqueeze_(0)

    return dataset

def eval_model2_ver2(model, rbf, df, data_name):
    
    dataset = val_missing_data_rbf2(df,rbf)
    dataset = dataset.to(device)

    real = pd.read_csv("./dataset/" + data_name).drop(['date'], axis = 1)
    real_scaler = (real - df.mean()) / df.std()

    df_scaler = ((df-df.mean()) / df.std()).values
    masks = ~np.isnan(df_scaler)
    masks = torch.from_numpy(masks).to(torch.float32)
    
    eval_masks = ~np.isnan(real_scaler.values)
    eval_masks = torch.from_numpy(eval_masks).to(torch.float32)

    test_masks = eval_masks - masks
    real_scaler = torch.nan_to_num(torch.from_numpy(real_scaler.values).to(torch.float32))
    
    model.eval()
    loss, x_loss, back_x_loss, loss_c, bi_c_hat, bi_imputation = model(dataset)

    Nonscale_imputataion = pd.DataFrame(bi_c_hat[0].cpu().detach() , columns= df.columns)
    Nonscale_imputataion = (Nonscale_imputataion * df.std()) + df.mean()
    
    real = real.fillna(0)
    print("Scale MAE :", torch.sum(torch.abs(bi_c_hat[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(test_masks))
    print("Scale MRE :", torch.sum(torch.abs(bi_c_hat[0].cpu().detach() - real_scaler) * test_masks) / torch.sum(torch.abs(real_scaler * test_masks)))

    print("Original MAE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(test_masks.cpu().numpy()))
    print("Original MRE :", np.sum(np.abs((Nonscale_imputataion - real).values * test_masks.cpu().numpy())) / np.sum(np.abs(real.values * test_masks.cpu().numpy())))

    print('Train MAE :', np.sum(np.abs((Nonscale_imputataion - real).values * masks.cpu().numpy())) / np.sum(masks.cpu().numpy()))
    print("Train MRE :", np.sum(np.abs((Nonscale_imputataion - real).values * masks.cpu().numpy())) / np.sum(np.abs(real.values * masks.cpu().numpy())))

    return Nonscale_imputataion