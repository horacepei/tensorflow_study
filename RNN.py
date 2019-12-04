# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
#配置环境
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
#初始化x和y的placeholder
X = tf.placeholder(tf.float32, [None, 784], name='X_placeholder')
Y = tf.placeholder(tf.int32, [None, 10], name='Y_placeholder')
# 初始化 w 权重 和 b 偏移
n_hidden_1 = 256 #隐藏层1
n_hidden_2 = 256 #隐藏层2
n_input = 784 #mnist 数据输入（28*28）
n_classes = 10 #mnist 10个首页数字类别

weights = {
        "h1":tf.Variable(tf.random_normal(shape=[n_input,n_hidden_1]), name='W1'),
        "h2":tf.Variable(tf.random_normal(shape=[n_hidden_1,n_hidden_2]), name='W2'),
        "out":tf.Variable(tf.random_normal(shape=[n_hidden_2,n_classes]), name='W')
        }

biases = {
        "b1":tf.Variable(tf.random_normal(shape=[n_hidden_1]), name='b1'),
        "b2":tf.Variable(tf.random_normal(shape=[n_hidden_2]), name='b2'),
        "out":tf.Variable(tf.random_normal(shape=[n_classes]), name='bias')
        }
# 构建graph 多层感知器
def multilayer_perceptron(x, weights, biases):
    #隐藏层1，激活函数relu
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'], name='fc_1')
    layer_1 = tf.nn.relu(layer_1, name='relu_1')
    #隐藏层2，激活函数
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'], name='fc_2')
    layer_2 = tf.nn.relu(layer_2, name='relu_2')
    #输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name='fc_3')
    return out_layer

# 预测类别score
pred = multilayer_perceptron(X, weights, biases)
    
# 计算损失函数并初始化optimizer 
# 求交叉熵的函数为损失函数
loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy')
# 求平均值
loss = tf.reduce_mean(loss_all, name='avg_loss')
# 初始化optimizer （优化器）
# 学习率
learning_rate = 0.01
# 使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
# 指定迭代次数
train_number = 50
# 每次取数据量
batch_size = 128
# 展示频度控制
display_step = 2
# 定义初始化全部变量op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化全部变量
    sess.run(init)
    # sess graph保存
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #开始执行时间
    start_time = time.time()

    #训练模型
    for i in range(train_number):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #遍历batchs
        for j in range(total_batch):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss],feed_dict={X:X_batch,Y:Y_batch})
            avg_loss += loss_batch/total_batch
        if train_number%display_step == 0:
            print("Average loss epoch {0}:{1:.9f}".format(i, avg_loss))
    print("Total time:{0} seconds".format(time.time() - start_time))
    print("Optimization Finished!")  
    
    #测试集测试
    #
    correct_preds = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(Y, 1))
    #
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
  
    print("Accuracy: {0}".format(accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))) 
    #关闭writer
    writer.close()
   