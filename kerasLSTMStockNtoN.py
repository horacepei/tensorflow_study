# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:17:06 2020

@author: 64054
"""
import os
import sys
#数据预处理以及绘制图形需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM, BatchNormalization
###############################################################################
def get_dataset():
    #从csv读取数据
    dataset = pd.read_csv('./000001.csv', encoding = 'gb18030')
    #print(dataset)
    #数据集的维度
    print(dataset.shape)
    #将数据价转换为整数
    dataset['开盘价'] = dataset['开盘价'].astype(int)
    dataset['收盘价'] = dataset['收盘价'].astype(int)
    # 列名选取行，剔除不需要开头的行
    dataframeDataset = dataset.loc[1:, ['开盘价', '收盘价']]
    print(dataframeDataset.shape)
    # 反转数据
    dataframeDatasetReverse = dataframeDataset.reindex(index = dataframeDataset.index[::-1])
    # 历史日期在前
    ndarrayDataset = dataframeDatasetReverse.values
    #print(ndarrayDataset)
    #画图大盘数据图
    #plt.plot(ndarrayDataset[:,0], ndarrayDataset[:,1])
    #plt.show()
    return ndarrayDataset

def get_lstm_dataset(dataset, need_num, total_dataset):
    lstm_dataset = []
    for i in range(need_num, total_dataset):
        lstm_dataset.append(dataset[i-need_num:i])
    lstm_dataset = np.array(lstm_dataset)
    #print(xTrain)
    #print(lstm_dataset.shape)
    return lstm_dataset

###############################################################################
#需要之前5次的股票数据来预测下一次的数据，
need_num = 5
#训练数据的大小
training_num = 240
#测试数据的大小
test_num = 1
#迭代训练10次
epoch = 10
#每次取数据数量
batch_size = 10
###############################################################################
#数据处理
ndarrayDataset = get_dataset()
#print(ndarrayDataset)
###############################################################################
# 构建训练集
x_train = ndarrayDataset[0:training_num]
#print(x_train)
y_train =  ndarrayDataset[1:training_num+1]
#print(y_train)
# 转换dt数据
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)
print(y_train.shape)
###############################################################################
#数据归一化,fit_transform处理后就变成list
sc_X = MinMaxScaler(feature_range=(0, 1))
x_train = sc_X.fit_transform(x_train)
x_train = np.array(x_train)
print(x_train.shape)

#print(x_train)
#print(x_train.shape)
# outcome scaling:
sc_Y = MinMaxScaler(feature_range=(0, 1))
y_train = sc_Y.fit_transform(y_train)
y_train = np.array(y_train)
print(y_train.shape)

###############################################################################
print('=======================数据转换=======================')
xTrain = []
for i in range(need_num, training_num):
    xTrain.append(x_train[i-need_num:i])
xTrain = np.array(xTrain)
#print(xTrain)
print(xTrain.shape)
x_train = xTrain

yTrain = []
for i in range(need_num, training_num):
    yTrain.append(y_train[i])
yTrain = np.array(yTrain)
#print(yTrain)
print(yTrain.shape)
y_train = yTrain
#因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, 1]，因此对数据进行相应转化
#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

###############################################################################
#构建网络，使用的是序贯模型
model = Sequential()
#return_sequences=True返回的是全部输出，LSTM做第一层时，需要指定输入shape
# step 1次取多少行数据
print(x_train.shape[1])
# 特征值数量 
print(x_train.shape[2])
model.add(LSTM(units=128, return_sequences=True, 
               input_shape=[x_train.shape[1],
                            x_train.shape[2]]))
model.add(BatchNormalization())
 
model.add(LSTM(units=128))
model.add(BatchNormalization())
 
model.add(Dense(units=x_train.shape[2]))
#进行配置
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
###############################################################################
#训练网络
model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
###############################################################################
#进行测试数据的处理
#将训练集的数据向后移动，增加上需要预测的数据
x_test = ndarrayDataset[test_num:training_num+test_num]
y_test = ndarrayDataset[test_num+1:training_num+test_num+1]
#print(y_train)
# 转换dt数据
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)
###############################################################################
#测试数据的归一化
sc_X_T = MinMaxScaler(feature_range=(0, 1))
x_test = sc_X_T.fit_transform(x_test)
x_test = np.array(x_test)
#print(x_test)
print(x_test.shape)
sc_Y_T = MinMaxScaler(feature_range=(0, 1))
y_test_transform = sc_Y_T.fit_transform(y_test.reshape(-1,1))
y_test_transform = np.array(y_test_transform)
#print(y_test_transform)
print(y_test_transform.shape)
#获取对应的开盘价列
###############################################################################
#构建LSTM需要的数据
xTest = []
for i in range(need_num, training_num):
    xTest.append(x_test[i-need_num:i])
xTest = np.array(xTest)
#print(xTrain)
print(xTest.shape)
x_test = xTest

yTest = []
for i in range(need_num, training_num):
    yTest.append(y_test[i])
yTest = np.array(yTest)
#print(yTrain)
print(yTest.shape)
y_test = yTest
###############################################################################
#进行预测
y_predictes = model.predict(x=x_test)
print(y_predictes)
print(y_predictes.shape)
#使用 sc.inverse_transform()将归一化的数据转换回原始的数据，以便我们在图上进行查看
y_predictes = sc_Y_T.inverse_transform(X=y_predictes)
print(y_predictes)
###############################################################################
#绘制数据图表，红色是真实数据，蓝色是预测数据
plt.plot(y_test[:,0], color='red', label='Real open')
plt.plot(y_test[:,1], color='yellow',label='Real close')
plt.plot(y_predictes[:,0], color='blue', label='Predicted open')
plt.plot(y_predictes[:,1], color='green', label='Predicted close')
plt.title(label='ShangHai open/close Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='ShangHai open/close')
plt.legend()
plt.show()



