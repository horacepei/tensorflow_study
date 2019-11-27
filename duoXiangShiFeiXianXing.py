# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#训练数据和测试数据
number = 100
list_x = np.linspace(-4, 4, number)
list_y = np.sin(list_x)+np.random.uniform(-0.5, 0.5, number)

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
#多项式建模，建立op
Y_pred = tf.add(tf.multiply(X,W),b)
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
Y_pred = tf.add(tf.multiply(tf.pow(X,2),W2),Y_pred)
W3 = tf.Variable(tf.random_normal([1]), name='weight3') 
Y_pred = tf.add(tf.multiply(tf.pow(X,3),W3),Y_pred)
#初始化数据样本个数
n_samples = len(list_x)
#计算损失函数值
loss = tf.reduce_sum(tf.pow(Y_pred-Y, 2))/n_samples
#初始化optimizer （优化器）
#学习率
learning_rate = 0.01
#选择优化器，这里使用梯度下降算法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#指定迭代次属，并在session中执行graph
train_number = 1000
#定义初始化全部变量op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    #初始化全部变量
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #训练模型
    for i in range(train_number):
        total_loss = 0
        for x, y in zip(list_x, list_y):
            #通过feed_dict把数据喂进去 x,y in [(x1,y1),(x2,y2),...]
            #这里说明下先执行了 optimier op 然后再执行 loss op 使用的数据都是feed_dict的
            sess.run([optimizer, loss],feed_dict={X:x,Y:y})
        #每100次输出一个训练情况
        if i %100==0:
            print("step %d eval loss is %f" % (i, total_loss/n_samples))
    #关闭writer
    writer.close()
    #输出下最红的w和b 
    W = sess.run(W)
    W2 = sess.run(W2)
    W3 = sess.run(W3)
    b = sess.run(b)
    print(W)
    print(W2)
    print(W3)
    print(b)

plt.plot(list_x, list_y, 'bo', label='real data')
plt.plot(list_x, W*list_x+W2*np.power(list_x,2)+W3*np.power(list_x,3)+b, 'r', label='predicted data')
plt.legend()
plt.show()

