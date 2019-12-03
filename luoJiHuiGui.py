# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

#准备训练数据和测试数据
mnist = input_data.read_data_sets('D:\study_project\\tensorflow', one_hot=True)

#查看训练集数据维度
print(mnist.train.images.shape)
#查看训练集目标维度
print(mnist.train.labels.shape)
#查看测试集数据维度
print(mnist.test.images.shape)
#查看测试机目标维度
print(mnist.test.labels.shape)
###################################建模##################################
#初始化x和y的占位符
batch_size = 128
X = tf.placeholder(tf.float32, [batch_size, 784], name='X')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y')
# 初始化 w 和 b
W = tf.Variable(tf.random_normal(shape=[784,10]), name='weight')
b = tf.Variable(tf.zeros([1,10]), name='bias')
#定义op， 矩阵乘法
logits = tf.matmul(X, W) + b
#求交叉熵的函数为损失函数
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
#求平均值
loss = tf.reduce_mean(entropy)
#初始化optimizer （优化器）
#学习率
learning_rate = 0.01
#使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
#指定迭代次数
train_number = 50
#定义初始化全部变量op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 图结构
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #开始执行时间
    start_time = time.time()
    # 初始化全部变量
    sess.run(init)
    #计算分多少批拿数据
    n_batches = int(mnist.train.num_examples/batch_size)
    #训练模型
    for i in range(train_number):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss],feed_dict={X:X_batch,Y:Y_batch})
            total_loss += loss_batch
        print("Average loss epoch {0}:{1}".format(i, total_loss/n_batches))
    print("Total time:{0} seconds".format(time.time() - start_time))
    print("Optimization Finished!")    
    #关闭writer
    writer.close()

#测试模型 
with tf.Session() as sess:
    # 初始化全部变量
    sess.run(init)
    #
    preds = tf.nn.softmax(logits)
    #
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.arg_max(Y, 1))
    #
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    #
    n_batches = int(mnist.test.num_examples/batch_size)
    #
    total_correct_preds = 0  
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy],feed_dict={X:X_batch,Y:Y_batch})
        total_correct_preds += accuracy_batch[0]
    print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))    