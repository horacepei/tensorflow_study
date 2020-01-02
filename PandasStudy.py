# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:17:06 2020

@author: 64054
"""

#数据预处理以及绘制图形需要的模块
import numpy as np
import pandas as pd

#从csv读取数据
dataset = pd.read_csv('./000001.csv', encoding = 'gb18030')
print(dataset)
#在打开文件的时候，不显示列名行
dataset = pd.read_csv('./000001.csv', encoding = 'gb18030',header=1)
print(dataset)
#在打开文件的时候不显示左侧的编号列
dataset = pd.read_csv('./000001.csv', encoding = 'gb18030',index_col=0)
print(dataset)
#在打开文件的时候只取指定的列
dataset = pd.read_csv('./000001.csv', encoding = 'gb18030',usecols=[0,2])
print(dataset)
#使用 columns 方法，打印列名，作用是可以看到每一列的数据代表的含义
print('打印列名')
print(dataset.columns)
#使用 shape 方法，打印数据的维度，一共有几行几列
print('打印数据的维度')
print(dataset.shape)
#使用 loc[ ] 方法，打印第几个数据，参数是数据的行号，例如取第 1 行数据
print('第几个数据')
print(dataset.loc[0])
#该方法也可以类似于python中列表的操作，使用切片，例如取第 3 - 5 行的数据
print('第3 - 5 行数据')
print(dataset.loc[3:5])
#可以传进去一个列表，例如打印 2，3，5 行的数据
print('打印 2，3，5 行的数据')
print(dataset.loc[[2,3,5]])
#基于索引列来获取行
print('第几个数据')
print(dataset.iloc[1])
#该方法也可以类似于python中列表的操作，使用切片，例如取第 3 - 5 行的数据
print('第3 - 5 行数据')
print(dataset.iloc[3:5])
#可以传进去一个列表，例如打印 2，3，5 行的数据
print('打印 2，3，5 行的数据')
print(dataset.iloc[[2,3,5]])
#获取列
#基于列名来获取列
print('第日期列数据')
print(dataset['日期'])
#该方法也可以类似于python中列表的操作，使用切片，例如用列名的数据
print('第日期，股票代码，名称列数据')
print(dataset[['日期','股票代码','名称']])
#组合行列的选取
# 全部行的，指定列
print(dataset.loc[:, ['日期', '股票代码','名称']])
# 索引选取行
print(dataset.iloc[5:10, 0:3])
# 列名选取行
print(dataset.loc[5:10, ['日期', '股票代码','名称']])


