import pandas as pd
import torch
import numpy as np
import config as config


device = 'cuda'
print("Using:", device)

data = pd.read_csv('converted_data.csv')
room_temp = data['room_temp'].values / 30.0
temp_sup  = data['temp_sup'].values  / 40.0
airflow   = data['airflow'].values   

pos_tem = room_temp[1:]
cur_tem = room_temp[:-1]
tem_sup = temp_sup[:-1]
airflow = airflow[:-1]

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)

def data_split(Ns, Ne, x1, x2, x3, y):
    return (to_tensor(x1[Ns:Ne]).view(-1,1), to_tensor(x2[Ns:Ne]).view(-1,1),
            to_tensor(x3[Ns:Ne]).view(-1,1), to_tensor(y[Ns:Ne]).view(-1,1))


def get_data(Ns_tr, Ne_tr, Ns_cal, Ne_cal, Ns_t, Ne_t):
    cur_tem_tr, tem_sup_tr, airflow_tr, pos_tem_tr = data_split(Ns_tr, Ne_tr, cur_tem, tem_sup, airflow, pos_tem)
    cur_tem_cal, tem_sup_cal, airflow_cal, pos_tem_cal = data_split(Ns_cal, Ne_cal, cur_tem, tem_sup, airflow, pos_tem)
    cur_tem_t, tem_sup_t, airflow_t, pos_tem_t = data_split(Ns_t, Ne_t, cur_tem, tem_sup, airflow, pos_tem)

    X_tr = torch.cat((cur_tem_tr, tem_sup_tr, airflow_tr), dim=1)
    y_tr = pos_tem_tr
    X_t = torch.cat((cur_tem_t, tem_sup_t, airflow_t), dim=1)
    y_t = pos_tem_t

    X_cal = torch.cat((cur_tem_cal, tem_sup_cal, airflow_cal), dim=1)
    y_cal = pos_tem_cal
    return X_tr, y_tr, X_cal, y_cal, X_t, y_t

def norm_temp(x):
    return (x/config.norm_temp_constant).view(-1,1)

def norm_sup(x):
    return (x/config.norm_sup_constant).view(-1,1)