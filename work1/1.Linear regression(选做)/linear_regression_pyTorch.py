#!/usr/bin/python
# encoding:utf-8
"""
@author: lance
@version: 1.0.0
@license: Apache Licence
@file: linear_regression_pyTorch.py
@time: 2021/7/17 17:43
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def data_process(array):
    x_list = []
    y_list = []
    for i in range(0, len(array), 18):
        for j in range(24 - 9):
            mat = array[i + 9, j:j + 9]
            label = array[i + 9, j + 9]  # 第10行是PM2.5
            x_list.append(mat)
            y_list.append([label])
    x = np.array(x_list, dtype='float32')
    y = np.array(y_list, dtype='float32')
    return x, y


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # x = self.hidden(x)
        x = self.predict(x)
        return x


def train(net, train_ds, optimizer, loss_func, epochs):
    for epoch in range(epochs):
        flag = 1
        # print("epoch {}".format(epoch))
        for x, y in train_ds:
            prediction = net(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()  # 梯度置零
            loss.backward()  # 反向传播
            optimizer.step()  # 计算梯度

            if epoch % 100 == 0 and flag:
                flag = 0
                print('epoch {:4}/{} Loss={:.4}'.format(epoch, epochs, loss.item()))

def main():
    csv = pd.read_csv('train.csv', usecols=range(3, 27))
    # print(csv)
    csv = csv.replace(['NR'], [0.0])
    print(csv)
    array = np.array(csv).astype(float)
    x, y = data_process(array)
    x_train, y_train = torch.from_numpy(x[:3200]), torch.from_numpy(y[:3200])
    train_ds = TensorDataset(x_train, y_train)
    batch_size = 3200
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    x_val, y_val = x[3200:], y[3200:]
    print(x_train)
    print(y_train)
    epoch = 2000
    net = Net(9, 9, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    loss_func = torch.nn.MSELoss()  # 均方差
    train(net, train_dl, optimizer, loss_func, epoch)
    print(net(x_train[0]))
    print(y_train[0])


if __name__ == '__main__':
    main()
