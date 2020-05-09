#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# input_path下有多个文件，每一个文件为一个数据的聚类结果
# 文件的每一行: 原始数据 属于每个类别的概率
input_path = sys.argv[1]

plt.figure()

names = os.listdir(input_path)
plot_num = 1
for n in names:
    file_name = os.path.join(input_path, n)
    origin_datas = []
    probabilities = []
    prob_size = 0
    with open(file_name, 'r') as f:
        for line in f:
            if len(line) == 0 or line[0] == '#':
                continue
            data = [float(x) for x in line.strip().split()]
            origin_data = data[0:2]
            prob = data[2:]
            origin_datas.append(origin_data)
            probabilities.append(prob)
            prob_size = len(prob)

    print("origin_datas size: ", len(origin_datas))
    print("probabilities size: ", len(probabilities))

    plot_origin_data = np.array(origin_datas)
    plot_prob = np.array(probabilities)

    print("plot_origin_data.shape: ", plot_origin_data.shape)
    print("plot_prob.shape: ", plot_prob.shape)

    plt.subplot(1, len(names), plot_num)
    plt.title(n, size=18)
    for n in range(plot_origin_data.shape[0]):
        # find class
        class_id = 0
        class_prob = 0
        for id in range(prob_size):
            if plot_prob[n, id] > class_prob:
                class_id = id
                class_prob = plot_prob[n, id]

        color = ['#377eb8', '#ff7f00', '#4daf4a']
        plt.scatter(plot_origin_data[n, 0], plot_origin_data[n, 1], s=10, color=color[class_id], alpha=class_prob, linewidths=0)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

plt.show()