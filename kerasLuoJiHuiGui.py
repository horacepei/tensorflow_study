# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
#配置环境
import os
import sys
import time
import numpy as np
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
#解决reuse问题，可以反复的执行
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#查看训练集数据维度
print(train_images.shape)
#查看训练集目标维度
print(len(train_labels))
#查看测试集数据维度
print(test_images.shape)
#查看测试机目标维度
print(len(test_labels))
#预处理数据
"""
数据进行预处理，将其变换为网络要求的形状，并缩放到所有值都在 [0, 1] 区间。训练图像保
存在一个 uint8 类型的数组中，其形状为 (60000, 28, 28)，取值区间为 [0, 255]。需要将其
变换为一个 float32 数组，其形状为 (60000, 28 * 28)，取值范围为 0~1。
"""
train_images=train_images.reshape(60000, 28 * 28)
train_images=train_images.astype('float32')/255
test_images=test_images.reshape(10000, 28 * 28)
test_images=test_images.astype('float32')/255
#标签进行分类编码。将类别向量转换为二进制（只有0和1）
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels)
print(test_labels)

###################################建模##################################
#设置全局参数
##
n_batch_size = 128
##训练代数，
n_epochs = 10

#构建网络
"""
 2 个 Dense 层，密集连接（也叫全连接）的神经层。
 第二层（也是最后一层）是一个 10 路 softmax 层，它将返回一个由 10 个概率值（总和为 1）组成的数组
""" 
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


"""
优化器（optimizer）：基于训练数据和损失函数来更新网络的机制。
损失函数（loss function）：网络如何衡量在训练数据上的性能，即网络如何朝着正确的方向前进。
在训练和测试过程中需要监控的指标（metric）：本例只关心精度，即正确分类的图像所占的比例。
"""
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#训练网络,这里要使用tensorboard
network.fit(train_images, train_labels, 
            epochs=n_epochs, batch_size=n_batch_size,
            callbacks=[TensorBoard(log_dir='./graphs')])

#测试数据(精度)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

