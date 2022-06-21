#读取数据,并进行增广，并打乱，然后创建模型训练需要的迭代器
#这里有个问题，迭代的样本都是excel，每次打开excel会很慢，而且打开excel只为了读取一列数据效率很低

import os
import random
from processing import data_process_1,data_process
from torch.utils.data import Dataset
import numpy as np
from feature import get_feature, get_feature_1, get_feature_2
import matplotlib.pyplot as plt

# 柔性传感器数据集
#def sensor_data_path(filepath,ratio,fix_index):
def sensor_data_path(filepath,ratio):
    # filepath是数据集的根路径，ratio是划分比例
    # 返回路径列表
    people = ["hy", "lzh", "lsy", "mxy", "tl"]
    train_s, train_l, test_s, test_l = [],[],[],[]
    #train_set,train_label,test_set,test_label = [],[],[],[]
    dirs1 = os.listdir(filepath)
    e = int(ratio*10)
    words = ["one.xlsx"]

    labsIndName = []  # 人名
    for p in dirs1:
        #if p != "zwl":continue
        #if (p in people): continue
        #print("loading ",p)
        dirs2 = os.listdir(filepath + "\\" + p + "\\" + "sensor")
        labsIndName.append(p)
        label = labsIndName.index(p)
        for i in dirs2:
            #if not (i in words): continue
            t = filepath + "\\" + p + "\\" + "sensor" + "\\" + i #路径名
            for j in range(e):
                train_s.append(t + "*" + str(j))
                train_l.append(label)
            for j in range(e,10): #这里的10表示每个excel有10个样本，即每个人说的每个单词有十遍,如果超过10，后续的切割字符串要检查
                test_s.append(t + "*" + str(j)) # 星号之后为数据标签，意为“第几条数据”，减号为增广标志，例如0为原始数据
                test_l.append(label)

    # 划分数据集,读取数据并增广
    print("loading......")
    train_set, train_label, test_set, test_label = [],[],[],[]
    l1 = len(train_s)
    l2 = len(test_s)
    for i in range(l1):
        for j in range(1):
            train_set.append(get_feature(data_process(train_s[i] + "-" + str(j))))
            train_label.append(train_l[i])  #记得改
            # plt.figure(0)
            # plt.xlabel('dimensionality')
            # plt.ylabel('value')
            # plt.plot(get_feature(data_process(train_s[i] + "-" + str(0))))
            # plt.show()
            # plt.figure(1)
            # plt.xlabel('dimensionality')
            # plt.ylabel('value')
            # plt.plot(get_feature(data_process(train_s[i] + "-" + str(1))))
            # plt.show()
            # plt.figure(2)
            # plt.xlabel('dimensionality')
            # plt.ylabel('value')
            # plt.plot(get_feature(data_process(train_s[i] + "-" + str(3))))
            # plt.show()
            # plt.figure(3)
            # plt.xlabel('dimensionality')
            # plt.ylabel('value')
            # plt.plot(get_feature(data_process(train_s[i] + "-" + str(5))))
            # plt.show()

    for i in range(l2):
        for j in range(1):
            test_set.append(get_feature(data_process(test_s[i] + "-" + str(j))))
            test_label.append(test_l[i])  #记得改
    print("train_set: ", len(train_set), " test_set: ", len(test_set))
    return train_set,train_label,test_set,test_label

# 自制麦克风数据集
def my_mic_data_path(filepath,ratio):
    # filepath是数据集的根路径，ratio是划分比例
    # 返回路径列表
    data_set,data_label = [],[]
    #train_set,train_label,test_set,test_label = [],[],[],[]
    people = ["hy", "lzh", "lsy", "mxy", "tl"]
    words = ["one", "zero", "three", "six"]
    dirs1 = os.listdir(filepath)
    for p in dirs1:
        #if p != "zwl": continue
        #if (p in people): continue
        print("loading ", p)
        dirs2 = os.listdir(filepath + "\\" + p + "\\" + "mic")
        labsIndName = []  #标签的名字，例如["one","two"]
        for i in dirs2:
            #if not (i in words): continue
            labsIndName.append(i)
            t = filepath + "\\" + p + "\\" + "mic" + "\\" + i #路径名
            label = labsIndName.index(i)
            dirs3 = os.listdir(t)
            #打标签，增广
            for j in dirs3:
                temp = t + "\\" + j
                for k in range(1):
                    data_set.append(get_feature_1(data_process_1(temp + "-" + str(k)), rate=16000)) #减号后为增广标志，例如0为原始数据
                    data_label.append(label) # 打标签
        #print(labsIndName)  # 检查标签顺序
    # 打乱数据集
    fix_index = np.arange(len(data_set))
    np.random.shuffle(fix_index)
    data_set = np.array(data_set)[fix_index]
    data_label = np.array(data_label)[fix_index]
    # 划分数据集
    a = data_set.shape[0]
    b = int(a * ratio)
    train_set = data_set[0:b]
    train_label = data_label[0:b]
    test_set = data_set[b:a]
    test_label = data_label[b:a]
    print("train_set: ",len(train_set)," test_set: ",len(test_set))
    return train_set,train_label,test_set,test_label,fix_index
    #return train_set, train_label, test_set, test_label

#公开数据集
def mic_data_path(filepath,ratio):
    # filepath是数据集的根路径，ratio是划分比例
    # 返回路径列表
    data_set, data_label = [], []
    dirs1 = os.listdir(filepath)
    test = ["one","two"]
    labsIndName = []  #标签的名字，例如["one","two"]
    for p in dirs1:
        #if p != "down": continue
        #if not (p in test): continue
        print("loading ", p)
        dirs2 = os.listdir(filepath + "\\" + p)
        labsIndName.append(p)
        for i in dirs2:
            t = filepath + "\\" + p + "\\" + i #路径名
            label = labsIndName.index(p)
            # data_set.append(get_feature(t + "-" + str(0), rate=16000))
            # data_label.append(label)
            # 打标签，增广
            for k in range(1):
                data_set.append(get_feature(t + "-" + str(k), rate=16000)) #减号后为增广标志，例如0为原始数据
                data_label.append(label) # 打标签
    #print(labsIndName)  # 检查标签顺序
    # 打乱数据集
    index = np.arange(len(data_set))
    np.random.shuffle(index)
    data_set = np.array(data_set)[index]
    data_label = np.array(data_label)[index]
    # 划分数据集
    a = data_set.shape[0]
    b = int(a * ratio)
    train_set = data_set[0:b]
    train_label = data_label[0:b]
    test_set = data_set[b:a]
    test_label = data_label[b:a]
    print("train_set: ",len(train_set)," test_set: ",len(test_set))
    return train_set,train_label,test_set,test_label

class MyDataset(Dataset):
    #初始化
    def __init__(self, data_set=None, data_label=None):
        self.dataset = data_set
        self.target = data_label
        #self.rate = rate
        #self.transforms = transforms
    #获取一个向量和标签
    def __getitem__(self, index):
        x = self.dataset[index]
        y = self.target[index]
        return x, y
    #获取长度
    def __len__(self):
        return len(self.dataset)

class parallel_dataset(Dataset):
    #初始化
    def __init__(self, data_set=None, data_label=None, transforms=None):
        self.dataset = data_set
        self.label = data_label
        #self.transforms = transforms
    #获取一个向量和标签
    def __getitem__(self, index):
        x = self.dataset[index]
        x1,x2 = x[0],x[1]
        y = self.label[index]
        return x1,x2,y
        #return x, y
    #获取长度
    def __len__(self):
        return len(self.dataset)