import numpy as np
from numpy import random
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns
sns.set()

def drawHeatmapFromFile(file_name):
    with open(file_name, 'r') as f:
        data = []
        for line in f:
            if len(line) < 1:
                continue
            tmp = []
            for x in line.strip().split(' '):
                tmp.append(float(x))
            data.append(tmp)
        fig = plt.figure()
        sns_plot = sns.heatmap(data, cmap='viridis',              # BuPu viridis plasma cividis
            xticklabels=False, yticklabels=False, annot=False)
        plt.show()

input = sys.argv[1]
drawHeatmapFromFile(input)
