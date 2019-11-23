# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#训练数据和测试数据
number=100
list_x = []
list_y = []

for i in range(number):
    x = np.random.randn(1)
    #这里构建的数据的分布满足 y=2*x+3 增加了一些噪点
    y = 2*x+3+np.random.randn(1)*1
    list_x.append(x)
    list_y.append(y)
print(list_x)
print(list_y)

plt.scatter(list_x, list_y)
plt.show()
###################################建模##################################
#初始化x和y的占位符
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
# 初始化 w 和 b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#建模，建立op y=w*x+b
Y_pred = tf.add(tf.multiply(X,W),b)
#计算损失函数值
loss = tf.square(Y-Y_pred, name='loss')
#初始化optimizer （优化器）
#学习率
learning_rate = 0.01
#选择优化器，这里使用梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#指定迭代次属，并在session中执行graph
#初始化数据样本个数
n_samples = len(list_x)
#初始化全部变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #训练模型
    for i in range(100):
        total_loss = 0
        for x, y in zip(list_x, list_y):
            #通过feed_dict把数据喂进去 x,y in [(x1,y1),(x2,y2),...]
            #这里说明下先执行了 optimier op 然后再执行 loss op 使用的数据都是feed_dict的
            sess.run([optimizer, loss],feed_dict={X:x,Y:y})
        #每10次输出一个训练情况
        if i %10==0:
            print("step %d eval loss is %f" % (i, total_loss/n_samples))
    #关闭writer
    writer.close()
    #输出下最红的w和b 
    W = sess.run(W)
    b = sess.run(b)
    print(W)
    print(b)

plt.plot(list_x, list_y, 'bo', label='real data')
plt.plot(list_x, W*list_x+b, 'r', label='predicted data')
plt.legend()
plt.show()
