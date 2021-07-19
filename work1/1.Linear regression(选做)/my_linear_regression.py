#!/usr/bin/python
# encoding:utf-8
"""
@author: lance
@version: 1.0.0
@license: Apache Licence
@file: my_linear_regression.py
@time: 2021/7/15 18:02
"""
import time

"""
根据作业要求可知，需要用到连续 9个时间点的气象观测数据，来预测第 10个时间点
的 PM2.5含量。针对每一天来说，其包含的信息维度为 (18,24)(18项指标， ，24个时间节点 )。
可以将 0到 8时的数据截取出来，形成一个维度为 (18,9)的数据帧，作为训练数据，将 9时
的 PM2.5含量取出来，作为该训练数据对应的 label；同理可取 1到 9时的数据作为训练用
的数据帧， 10时的 PM2.5含量作为 label......以此分割，可将每天的信息分割为 15个 shape为 (18,9)的数据帧和与之对应的 15个 label。
训练集中共包含 240天的数据，因此共可获得 240X15=3600个数据帧和与之对应的
3600个 label。
将前 2400个数据帧作为训练集，后 1200个数据帧作为验证集。
"""

import pandas as pd
import numpy as np

"""

18 X 24
0-9 10-18
"""


class Model:
    """
    y = x * W + b
    """
    learning_rate = 1
    epoch = 2000
    params = []

    def __init__(self):

        bias = 0
        weights = np.random.randn(9)
        self.params = [weights, bias]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params



def data_process(array):
    x_list = []
    y_list = []
    for i in range(0, len(array), 18):
        for j in range(24 - 9):
            mat = array[i:i + 18, j:j + 9]
            label = array[i + 9, j + 9]  # 第10行是PM2.5
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


def train(x_train, y_train, epochs):
    start_time = time.time()
    bias = 0  # 偏置值初始化
    weights = np.ones(9)  # 权重初始化
    learning_rate = 1.2  # 初始学习率
    reg_rate = 0.001 # 正则项系数
    bg2_sum = 0 # 用于存放偏置值的梯度平方和
    wg2_sum = np.zeros(9) # 用于存放权重的梯度平方和

    for epoch in range(epochs):
        # print('Epoch [{}/{}]'.format(epoch + 1, epochs))
        b_g = 0
        w_g = np.zeros(9)
        # 在所有数据上计算Loss_label的梯度
        for j in range(3200):
            b_g += 2 * (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += 2 * (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])
        # 求平均
        b_g /= 3200
        w_g /= 3200
        #  加上Loss_regularization在w上的梯度
        # for m in range(9):
        #     w_g[m] += reg_rate * weights[m]

        # adagrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        # 每训练200轮，输出一次在训练集上的损失
        if epoch % 200 == 0:
            loss = 0
            for j in range(3200):
                loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) ** 2
            print('after {} epochs, the loss on train data is:'.format(epoch), loss / 3200)

    return weights, bias


def validate(x_val, y_val, weights, bias):
    loss = 0
    for i in range(400):
        loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias)**2
    return loss / 400


def main():
    csv = pd.read_csv('train.csv', usecols=range(3, 27))
    # print(csv)
    csv = csv.replace(['NR'], [0.0])
    print(csv)
    array = np.array(csv).astype(float)
    x, y = data_process(array)
    x_train, y_train = x[:3200], y[:3200]
    x_val, y_val = x[3200:], y[3200:]
    epoch = 2000
    # print(len(y))
    w, b = train(x_train, y_train, epoch)
    print('{}'.format(w.dot(x_train[0, 9, :]) + b))
    print('{}'.format(y_train[0]))
    loss = validate(x_val, y_val, w, b)
    print('The loss on val data is:', loss)

if __name__ == '__main__':
    main()
