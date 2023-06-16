import random

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms



class TuneDataSet(Dataset):  # 继承Dataset
    def __init__(self, root,transform, is_train=True):  # __init__是初始化该类的一些基础参数
        self.root_dir = root  # 文件目录
        self.Y = []
        self.X = []
        self.count = 0
        self.transform = transform


    def __len__(self):  # 返回整个数据集的大小
        return self.count

    def __getitem__(self, index):
        target, label = self.X[index], self.Y[index]
        img = Image.open(self.root_dir+str(label)+"/"+target).convert('RGB')
        x = torch.cat([x1,x2,x3], dim=0)
        return x, label

# sort function

if __name__ == '__main__':
    root_dir = "/home/kali/ML/Jaundice-model/datasets/conflict/no_confict_small/"

    train_dataset = ColorSet(root_dir, True)
    train_dataset.__getitem__(1)
    print(len(train_dataset))


