# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:43:11 2019

@author: 64054
"""

import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt

print('Data -----------')
#构建训练数据和测试数据 （100个点）
number=100

list_x = []
list_y = []

for i in range(number):
    #返回[0,1)的随机x，不给维度就是数字，给维度就是返回对应维度的随机值
    x = np.random.randn()
    #这里构建的数据的分布满足y=2*x+3 增加了一些噪点
    y = 2*x+3+np.random.randn()*1
    list_x.append(x)
    list_y.append(y)
print(list_x)
print('\n')
print(list_y)

plt.scatter(list_x, list_y)
plt.show()

print('Model -----------')
 # 把前80个数据放到训练集
X_train, Y_train = list_x[:80], list_y[:80] 
# 把后20个点放到测试集   
X_test, Y_test = list_x[80:], list_y[80:]   

# 定义一个模型
# Keras 单输入单输出的线性序列模型 Sequential，
model = Sequential () 
#设置模型
#通过add()方法一层层添加模型，Dense是全连接层，第一层需要定义输入，
model.add(Dense(output_dim=1, input_dim=1)) 

#选择损失函数和优化器
model.compile(loss='mse', optimizer='sgd')
# 开始训练
print('Training -----------')
for step in range(200):
    cost = model.train_on_batch(X_train, Y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 20 == 0:
        print('train cost: ', cost)
# 查看训练出的网络参数：权重和偏移
W, b = model.layers[0].get_weights()   
print('Weights=', W, '\nbiases=', b)

# 测试训练好的模型
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=20)
print('test cost:', cost)
#预测值
Y_pred = model.predict(X_test)
#画图(蓝点，红线)
plt.scatter(X_test, Y_test, c='b', marker='o', label='real data')
plt.plot(X_test, Y_pred, c='r', label='predicted data')
plt.show()