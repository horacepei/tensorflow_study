# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:57:56 2019

@author: 64054
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential
# Dense全连接层
from keras.layers import Dense, Activation
# 优化器：随机梯度下降
from keras.optimizers import SGD

# 生成非线性数据模型
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#训练数据和测试数据
number = 100
x_data = np.linspace(-4, 4, number)
y_data = np.sin(x_data)+np.random.uniform(-0.5, 0.5, number)

print(x_data)
print(y_data)
# 显示随机点
plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = Sequential()

# 在模型中添加一个全连接层
# 神经网络结构：1-10-1，即输入层为1个神经元，隐藏层10个神经元，输出层1个神经元。 

# 激活函数加法1
model.add(Dense(units=10, input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(units=1))
model.add(Activation('tanh'))

# 激活函数加法2
# model.add(Dense(units=10, input_dim=1, activation='relu'))
# model.add(Dense(units=1, activation='relu'))

# 定义优化算法
sgd = SGD(lr=0.3)
# sgd: Stochastic gradient descent,随机梯度下降法
# mse: Mean Squared Error, 均方误差
model.compile(optimizer=sgd, loss='mse')

# 进行训练
for step in range(3000):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    # 每500个batch打印一次cost值
    if step % 200 == 0:
        print('cost: ', cost)
# 打印权值和偏置值
W, b = model.layers[0].get_weights()
print('W：', W, ' b: ', b)
print(len(model.layers))

# 把x_data输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data, y_data)
# 显示预测结果
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()