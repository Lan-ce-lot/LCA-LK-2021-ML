#!/usr/bin/python
# encoding:utf-8
"""
@author: lance
@version: 1.0.0
@license: Apache Licence
@file: LinearModel_numpy.py
@time: 2021/7/19 10:55
"""
import numpy as np

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

w = 1
b = 1


def forward(x):
    y_pred = predict(x)
    return y_pred


def predict(x):
    return x * w + b


def get_loss(y_pred, y_data):
    return (y_pred - y_data) ** 2


epoch = 100
for i in range(epoch):
    b_g = 0
    w_g = 0
    lr = 0.01
    loss = 0
    bg2_sum = 0 # 用于存放偏置值的梯度平方和
    wg2_sum = 0 # 用于存放权重的梯度平方和
    for x, y in zip(x_data, y_data):
        b_g += (predict(x) - y)
        w_g += (predict(x) - y) * x
        loss += (predict(x) - y) ** 2
    b_g /= len(x_data)
    w_g /= len(x_data)
    loss /= len(x_data)
    print(i, loss)
    # adagrad
    bg2_sum += b_g ** 2
    wg2_sum += w_g ** 2
    b -= lr * b_g / bg2_sum ** 0.5
    w -= lr * w_g / wg2_sum ** 0.5


print('w = ', w)
print('b = ', b)

x_test = 4.0
y_test = predict(4.0)
print('y_pred = ', y_test)


if __name__ == '__main__':
    pass
