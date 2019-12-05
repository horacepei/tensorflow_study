# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:56:46 2019

@author: 64054
"""
#配置环境
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import time
#解决reuse问题，可以反复的执行
tf.reset_default_graph() 
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
#设置全局参数
##学习率
learning_rate = 0.001
##迭代次数
training_iters = 100000
##每批拿多少张图片
batch_size = 128
##展示间隔
display_step = 10
#设置网络参数设置
##数据输入 每次28个
n_input = 28
##时序， 28次
n_steps = 28
##隐藏层
n_hidden = 128 
##最后药区分得类别 （0-9）一共10个数字
n_classes = 10
#初始化x和y的placeholder
X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X_placeholder')
Y = tf.placeholder(tf.int32, [None, n_classes], name='Y_placeholder')
# 初始化 w 权重 和 b 偏移
weights = {
        "out":tf.Variable(tf.random_normal(shape=[n_hidden,n_classes]), name='W')
        }

biases = {
        "out":tf.Variable(tf.random_normal(shape=[n_classes]), name='bias')
        }
# 构建RNN网络（这里使用相同cell）
def RNN(x, weights, biases):
    #为了适应rnn 调整原始输入数据,调整到维度相同
    inputs = tf.unstack(x, n_steps, axis=1)
    #定义一个lstm单元, 允许迭代同一个变量再一个命名空间
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    #输出层
    outputs, states = rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
    #取最后一个输出
    result = tf.matmul(outputs[-1], weights['out'])+biases['out']
    return result

# 预测类别score
pred = RNN(X, weights, biases)
# 计算损失函数并初始化optimizer 
##求交叉熵的函数为损失函数
cost_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y, name='cross_entropy')
##求平均值
cost = tf.reduce_mean(cost_all, name='avg_loss')
##初始化optimizer （优化器），使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

#评估模型
#
correct_preds = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(Y, 1))
#
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# 定义初始化全部变量op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化全部变量
    sess.run(init)
    # sess graph保存
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    #开始执行时间
    start_time = time.time()
    step = 1
    #训练模型
    while (step*batch_size) < training_iters:
        #分批取数据
        X_batch, Y_batch = mnist.train.next_batch(batch_size)
        #调整数据形态
        X_batch = X_batch.reshape((batch_size, n_steps, n_input))
        #优化器优化
        sess.run(optimizer, feed_dict={X:X_batch,Y:Y_batch})
        #显示训练情况
        if step % display_step == 0:
            #计算准确率
            acc = sess.run(accuracy, feed_dict={X:X_batch,Y:Y_batch})
            #计算batch上的loss
            loss = sess.run(cost, feed_dict={X:X_batch,Y:Y_batch})           
            print("Iter " + str(step*batch_size) \
               + "|Minibatch loss=" + "{:.6f}".format(loss) \
               + "|Training Accuracy=" + "{:.6f}".format(acc))
        step += 1
    print("Total time:{0} seconds".format(time.time() - start_time))
    print("Optimization Finished!")  
    #关闭writer
    writer.close()
   