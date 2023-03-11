### 生成用于模型的数据集
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DiabetesDataset(Dataset):
    def _init_(self,datapath):
        df=pd.read_csv(datapath, columns=['user_id', 'item_id', 'lable'])
        self.len=df.columns[0] ### 行数
        self.x_data=df[[df['user_id'],df[1]]]
        self.y_data=df[df[2]] ### 保证label和user id, item id为数值
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

### odps  数据平台，数据读取的代码
### 星云，模型运行平台