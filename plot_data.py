# -*- coding: utf-8 -*-
"""


"""

import sys

sys.path.insert(0, '../../Utilities/')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
import math
import tensorflow as tf
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import datetime
from pyDOE import lhs
from scipy.integrate import odeint
start_time = time.time()


# Load data
data_frame = pandas.read_csv('Data/BC_data.csv')
I1_new_star = data_frame['I1new']  # T x 1 array
I2_new_star = data_frame['I2new']  # T x 1 array
I1_sum_star = data_frame['I1sum']  # T x 1 array
I2_sum_star = data_frame['I2sum']  # T x 1 array
# 数据转化
I1_new_star = I1_new_star.to_numpy(dtype=np.float64)
I2_new_star = I2_new_star.to_numpy(dtype=np.float64)
I1_sum_star = I1_sum_star.to_numpy(dtype=np.float64)
I2_sum_star = I2_sum_star.to_numpy(dtype=np.float64)
# t_star = np.arange(len(I1_new_star))


first_date = np.datetime64('2023-03-26')
last_date = np.datetime64('2024-01-07') + np.timedelta64(7, 'D')
t_star = np.arange(first_date, last_date, np.timedelta64(7, 'D'))
# 存储路径
current_directory = os.getcwd()
relative_path_figs = '/figs_data/'
save_figs_to = current_directory + relative_path_figs


# 画图
SAVE_FIG = True
#################### I1 ##########################
# 创建条形图的宽度
bar_width = 0.35

# 生成条形图的位置
index = np.arange(len(t_star))

#自定义颜色
color1 = (237/255, 173/255, 197/255)  # 自定义颜色1
color2 = (108/255, 190/255, 195/255)  # 自定义颜色2


# 创建条形图
fig, ax = plt.subplots()
bars1 = ax.bar(index, I1_new_star, bar_width, label='New reported cases', color=color1)
bars2 = ax.bar(index + bar_width, I1_sum_star, bar_width, label='Cumulative reported cases', color=color2)

# 添加标题和标签
plt.title('BC')
plt.xlabel('Times')
plt.ylabel('The number of weekly reported cases of XBB.$1.16.^{*}$')
plt.xticks(index[::8] + bar_width / 2, t_star[::8])
# 添加图例
plt.legend(frameon=False,fontsize='large')


#保存图片
if SAVE_FIG:
    plt.savefig(save_figs_to + 'I1.png', dpi=300)
    plt.savefig(save_figs_to + 'I1.pdf', dpi=300)





#################### I2 ##########################
# 创建条形图的宽度
bar_width = 0.35

# 生成条形图的位置
index = np.arange(len(t_star))

#自定义颜色
color1 = (206/255, 170/255, 208/255)  # 自定义颜色1
color2 = (97/255, 156/255, 217/255)  # 自定义颜色2


# 创建条形图
fig, ax = plt.subplots()
bars3 = ax.bar(index, I2_new_star, bar_width, label='New reported cases', color=color1)
bars4 = ax.bar(index + bar_width, I2_sum_star, bar_width, label='Cumulative reported cases', color=color2)

# 添加标题和标签
plt.title('BC')
plt.xlabel('Times')
plt.ylabel('The number of weekly reported cases of XBB.1.5.')
plt.xticks(index[::8] + bar_width / 2, t_star[::8])
# 添加图例
plt.legend(frameon=False,fontsize='large')


#保存图片
if SAVE_FIG:
    plt.savefig(save_figs_to + 'I2.png', dpi=300)
    plt.savefig(save_figs_to + 'I2.pdf', dpi=300)

# 显示图形
plt.show()



