import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tqdm import tqdm
import math
import random
import numpy as np
import pandas as pd
import torch.optim as optim

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size).cuda() - torch.eye(input_size, input_size).cuda()
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size)).cuda()
        self.b = Parameter(torch.Tensor(output_size)).cuda()
        self.relu = nn.ReLU(inplace=False)
        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size).cuda()
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = self.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    
class MGRU_RBF(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGRU_RBF, self).__init__()

        self.temp_decay_h = TemporalDecay(input_size, output_size = hidden_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size, input_size, diag = True)
        self.temp_decay_r = TemporalDecay(input_size, input_size, diag = True)
        
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.build()

    def build(self):
        self.output_layer = nn.Linear(self.hidden_size, self.input_size, bias=True)
        
        self.z_layer = FeatureRegression(self.input_size)
        self.beta_layer = nn.Linear(self.input_size * 2, self.input_size)
        self.grucell = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.concat_lyaer = nn.Linear(self.input_size * 2, self.input_size)
        

    def loss(self, hat, y, m):
        return torch.sum(torch.abs((y - hat)) * m) / (torch.sum(m) + 1e-5)

    
    def forward(self, input):
        values = input[:,0,::]
        delta = input[:,1,::]
        masks = input[:,2,::]
        rbfs = input[:,3,::]

        hid = torch.zeros((values.size(0), self.hidden_size)).cuda()

        x_loss = 0.0
        x_hat_loss = 0.0
        concat_loss = 0.0
        z_hat_loss = 0.0 
        c_hat_loss = 0.0
        imputations = []
        c_hat_list = []
        for i in range(values.size(1)):

            v = values[:,i,:]
            d = delta[:,i,:]
            m = masks[:,i,:]
            r = rbfs[:,i,:]

            gamma_x = self.temp_decay_x(d)
            gamma_h = self.temp_decay_h(d)

            
            x_hat = self.output_layer(hid)
            x_loss += torch.sum(torch.abs(v - x_hat) * m) / (torch.sum(m) + 1e-5)
            x_hat_loss += torch.sum(torch.abs(v - x_hat) * m) / (torch.sum(m) + 1e-5)
            x_c = m * v + (1 - m) * x_hat

            RG = torch.cat([x_c, r], dim = 1)
            concat_hat = self.concat_lyaer(RG)
            x_loss += torch.sum(torch.abs(v - concat_hat) * m) / (torch.sum(m) + 1e-5)
            concat_loss += torch.sum(torch.abs(v - concat_hat) * m) / (torch.sum(m) + 1e-5)
            a_c = m * v + (1 - m) * concat_hat

            z_hat = self.z_layer(a_c)
            x_loss += torch.sum(torch.abs(v - z_hat) * m) / (torch.sum(m) + 1e-5)
            z_hat_loss += torch.sum(torch.abs(v - z_hat) * m) / (torch.sum(m) + 1e-5)

            beta_weight = torch.cat([gamma_x, m], dim = 1)
            beta = torch.sigmoid(self.beta_layer(beta_weight))

            c_hat = beta * z_hat + (1 - beta) * x_hat
            x_loss += torch.sum(torch.abs(v - c_hat) * m) / (torch.sum(m) + 1e-5)
            c_hat_loss += torch.sum(torch.abs(v - c_hat) * m) / (torch.sum(m) + 1e-5)

            c_c = m * v + (1 - m) * c_hat

            gru_input = torch.cat([c_c, m], dim = 1)
            imputations.append(c_c.unsqueeze(dim = 1))
            c_hat_list.append(c_hat.unsqueeze(1))
            
            # GRU cell
            hid = hid * gamma_h
            hid = self.grucell(gru_input, hid)

        c_hat_list = torch.cat(c_hat_list, dim = 1)
        imputations = torch.cat(imputations, dim = 1)
        return c_hat_list, imputations, x_loss * 5

class BiMGRU_RBF(nn.Module):
    def __init__(self, input_size, hidden_size, bias = True):
        super(BiMGRU_RBF, self).__init__()
        
        self.fmGRU = MGRU_RBF(input_size, hidden_size)
        self.bmGRU = MGRU_RBF(input_size, hidden_size)
    
    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss
    
    def backdirect_data(self, tensor_):
        if tensor_.dim() <= 1:
            return tensor_
    
        indices = range(tensor_.size()[2])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad = False)

        if torch.cuda.is_available():
            indices = indices.cuda()

        return tensor_.index_select(2, indices)
    
    def backdirect_imputation(self, tensor_):
        if tensor_.dim() <= 1:
            return tensor_
    
        indices = range(tensor_.size()[1])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad = False)

        if torch.cuda.is_available():
            indices = indices.cuda()

        return tensor_.index_select(1, indices)

    def forward(self, dataset):
        back_dataset = self.backdirect_data(dataset)

        c_hat_list, imputations, x_loss = self.fmGRU(dataset)
        back_c_hat_list, back_x_imputataions, back_x_loss = self.bmGRU(back_dataset)

        loss_c = self.get_consistency_loss(c_hat_list, self.backdirect_imputation(back_c_hat_list))
        loss = x_loss + back_x_loss + loss_c

        bi_c_hat = (c_hat_list +  self.backdirect_imputation(back_c_hat_list)) / 2
        bi_imputation = (imputations +  self.backdirect_imputation(back_x_imputataions)) / 2

        return  loss, x_loss, back_x_loss, loss_c, bi_c_hat, bi_imputation

    
def train_BiMGRU(model, lr, epochs, dataset, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    progress = tqdm(range(epochs))
    
    imputation_list = []
    loss_list = []
    model.to(device)

    for epoch in progress:
        batch_loss = 0.0
        batch_f_loss = 0.0
        batch_b_loss = 0.0
        batch_c_loss = 0.0

        for data in dataset:
            data = data.to(device)

            loss, x_loss, back_x_loss, loss_c, bi_chat, biimputataion = model(data)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            imputation_list.append(bi_chat)

            batch_loss += loss
            batch_f_loss += x_loss
            batch_c_loss += loss_c
            batch_b_loss += back_x_loss
            
        progress.set_description("loss: {}, f_MGRU loss : {}, b_MGRU loss : {}, consistency Loss : {}".format(batch_loss, batch_f_loss, batch_b_loss, batch_c_loss))
        loss_list.append(loss)

    return loss_list
