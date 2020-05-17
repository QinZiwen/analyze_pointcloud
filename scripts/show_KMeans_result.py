#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# input_path下有多个文件，每一个文件为一个数据的聚类结果
# 文件的每一行为一个类别，比如两类，文件就有两行，每一行为一类数据，如果是二维数据则：x y x y ...
input_path = sys.argv[1]

plt.figure()

names = os.listdir(input_path)
plot_num = 1
for n in names:
    file_name = os.path.join(input_path, n)
    cluster_data = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line) == 0 or line[0] == '#':
                continue
            data = [float(x) for x in line.strip().split()]
            x = data[::2]
            y = data[1::2]
            print('x size: ', len(x))
            print('y size: ', len(y))
            cluster_data.append(x)
            cluster_data.append(y)
    print(len(cluster_data))
    plt.subplot(1, len(names), plot_num)
    plt.title(n, size=18)
    plt.scatter(cluster_data[0], cluster_data[1], s=10, color="#377eb8")
    plt.scatter(cluster_data[2], cluster_data[3], s=10, color="#ff7f00")
    plt.scatter(cluster_data[4], cluster_data[5], s=10, color="#4daf4a")
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

plt.show()