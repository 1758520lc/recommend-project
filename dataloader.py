### 生成用于模型的数据集
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DiabetesDataset(Dataset):
    def _init_(self,datapath):
        df=pd.read_csv(datapath,)
        self.len=df.columns[0]
        self.x_data=torch.df[[columns[0],columns[1]]]
        self.y_data=torch.df[columns[2]]
    def _getitem_(self,index):
        return self.x_data[index],self.y_data[index]
    def _len_(self):
        return self.len
dataset1=DiabetesDataset('train_data_csv')
train_loader=DataLoader(dataset=dataset1,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)
dataset2=DiabetesDataset('test_data_csv')
test_loader=DataLoader(dataset=dataset2,
                       batch_size=32,
                       shuffle=True,
                       num_workers=2)

