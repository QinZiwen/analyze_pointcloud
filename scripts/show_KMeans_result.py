#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import math

# input_path下有多个文件，每一个文件为一个数据的聚类结果
# 文件的每一行为一个类别，比如两类，文件就有两行，每一行为一类数据，如果是二维数据则：x y x y ...
input_path = sys.argv[1]

plt.figure()

color = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
names = os.listdir(input_path)
rows = math.floor(len(names)/3) + len(names)%3
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
    print('len(cluster_data): ', len(cluster_data))
    plt.subplot(rows, 3, plot_num)
    plt.title(n, size=18)

    if len(cluster_data) / 2 < len(color):
        end = len(cluster_data)
    else:
        end = len(color) - 1
    print('end: ', end)
    for i in range(0, end, 2):
        print(i)
        plt.scatter(cluster_data[0 + i], cluster_data[1 + i], s=10, color=color[i/2])
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

plt.show()