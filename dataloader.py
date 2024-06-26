import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def train_test_datasets(data):
    X = data.drop('HI', axis=1, inplace=False)
    Y = data['HI']

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

    # transform to torch tensor
    train_tensor_x = torch.from_numpy(X_train.values).float() 
    train_tensor_y = torch.from_numpy(y_train.values).float()
    train_dataset = TensorDataset(train_tensor_x,train_tensor_y) 

    test_tensor_x = torch.from_numpy(X_test.values).float() 
    test_tensor_y = torch.from_numpy(y_test.values).float()
    test_dataset = TensorDataset(test_tensor_x,test_tensor_y)

    return train_dataset, test_dataset

def train_test_dataloaders(train_dataset, test_dataset, batch_size):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

