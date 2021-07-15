#!/usr/bin/python
# encoding:utf-8
"""
@author: lance
@version: 1.0.0
@license: Apache Licence
@file: my_linear_regression.py
@time: 2021/7/15 18:02
"""
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


def dataProcess(array):
    x_list = []
    y_list = []
    for i in range(0, len(array), 18):
        for j in range(24-9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9, j+9] # 第10行是PM2.5
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y

def main():
    csv = pd.read_csv('train.csv', usecols=range(3, 27))
    # print(csv)
    csv = csv.replace(['NR'], [0.0])
    print(csv)
    array = np.array(csv).astype(float)
    x, y = dataProcess(array)
    # for i in array:
    #     for j in i:
    #         print('{:6}'.format(j), end='')
    #     print()



if __name__ == '__main__':
    main()
