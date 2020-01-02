# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:33:56 2020

@author: 64054
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#数据处理
#从csv读取数据
dataset = pd.read_csv('./000001.csv', encoding = 'gb18030')
#print(dataset)
#数据集的维度
print(dataset.shape)
#将开盘价转换为整数
dataset['开盘价'] = dataset['开盘价'].astype(int)
# 列名选取行
dataframeDataset = dataset.loc[1:, ['日期', '开盘价']]
ndarrayDataset = dataframeDataset.values
#print(ndarrayDataset)
#画图大盘数据图
#plt.plot(narrayDataset[:,0], narrayDataset[:,1])
#plt.show()
#对大盘数据进行，数据归一化
stock_dataset = narrayDataset[:,1]
#显示部分数据 0-10条
print(stock_dataset[0:10])
#开盘价数据维度
print('reshap before:',stock_dataset, stock_dataset.shape)
#奖开盘价数据转换为nx1列的数据
stock_dataset = stock_dataset.reshape(-1,1)
#显示部分数据 0-10条
print('reshap after:', stock_dataset[0:10], stock_dataset.shape)
#初始化MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  
#转换
stock_dataset_transform = sc.fit_transform(stock_dataset)
#显示部分数据 0-10条
print('transformed:', stock_dataset_transform[0:10])
#反归一化，回复原有数据，显示部分数据 0-10条
stock_dataset_inverse = sc.inverse_transform(stock_dataset_transform)
print('inversed:', stock_dataset_inverse[0:10])