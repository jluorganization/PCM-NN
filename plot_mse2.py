import sys

sys.path.insert(0, '../../Utilities/')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas
import matplotlib.dates as mdates
import math
import tensorflow as tf
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

#################################  Load data ########################################


data_frame20202024 = pandas.read_csv('Data\Data_2020-2024.csv')
data_frame20202023 = pandas.read_csv('Data\Data_2020-2023.csv')
data_frame2024 = pandas.read_csv('Data\Data_2024.csv')
X = np.loadtxt('Model0924\Train-Results-09-24-set16\X.txt')
X1 = np.loadtxt('Model0924\Train-Results-09-24-set16\X1_ode.txt')
X24 = np.loadtxt('Model0924\Train-Results-09-24-set16\X24_ode.txt')

# 数据拟合时间长度
first_date = np.datetime64('2020-07-25')
last_date = np.datetime64('2020-08-23') + np.timedelta64(1, 'D')
t_star = np.arange(first_date, last_date, np.timedelta64(1, 'D'))

# 存储路径
current_directory = os.getcwd()
relative_path_figs = '/Data/'
save_figs_to = current_directory + relative_path_figs
SAVE_FIG = True


MSE_PCM = np.mean((X - data_frame20202023.iloc[:, 1]) ** 2)
MAE_PCM = np.mean(np.abs(X - data_frame20202023.iloc[:, 1]))
R2_PCM = 1 - (np.sum((X - data_frame20202023.iloc[:, 1]) ** 2) / (np.sum((X - np.mean(X)) ** 2)))

MSE_ODE = np.mean((X1 - data_frame20202023.iloc[:, 1]) ** 2)
MAE_ODE = np.mean(np.abs(X1 - data_frame20202023.iloc[:, 1]))
R2_ODE = 1 - (np.sum((X1 - data_frame20202023.iloc[:, 1]) ** 2) / (np.sum((X1 - np.mean(X1)) ** 2)))

MSE_pre24 = np.mean((X24 - data_frame2024.iloc[:, 1]) ** 2)
MAE_pre24 = np.mean(np.abs(X24 - data_frame2024.iloc[:, 1]))
R2_pre24 = 1 - (np.sum((X24 - data_frame2024.iloc[:, 1]) ** 2) / (np.sum((X24 - np.mean(X24)) ** 2)))

MSE_pre20202024 = np.mean((X24 - data_frame20202024.iloc[:, 1]) ** 2)
MAE_pre20202024 = np.mean(np.abs(X24 - data_frame20202024.iloc[:, 1]))
R2_pre20202024 = 1 - (np.sum((X24 - data_frame20202024.iloc[:, 1]) ** 2) / (np.sum((X24 - np.mean(X24)) ** 2)))
#

#                         ######################################################################
#                         ############################# Plotting ###############################
#                         ######################################################################
values1 = [R2_PCM, R2_ODE, R2_pre20202024]
values2 = [MSE_PCM , MSE_ODE, MSE_pre20202024]
values3 = [MAE_PCM , MAE_ODE, MAE_pre20202024]
y1_label = '$\t{R}^2$'
y2_label = '$\t{MSE}$'
y3_label = '$\t{MAE}$'
labels = ['PCM-NN Fitting', 'ODE Solver', 'Prediction VS Annual Average']
colors = [(241/255, 108/255, 35/255), (43/255, 106/255, 153/255), (27/255, 211/255, 223/255)]
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1,3, wspace=0.3, hspace=0.52)
# 创建一个固定的图例句柄大小（例如，高度为1的矩形）

handle_height = 1

handles = [plt.Rectangle((0, 0), 1, handle_height, fc=c) for c in colors]

# 第一个子图
ax11 = fig.add_subplot(gs[0, 0])
ax11.bar(range(len(values1)), values1, color=colors, label=[None] * len(values1))  # 不为这里的柱子设置label，因为我们在后面统一设置图例
ax11.set_ylabel(y1_label)
ax11.set_xlabel('Evaluative Aspects')
ax11.set_xticks([])  # 隐藏横坐标标签
ax11.legend(handles, labels, loc='best', frameon=False)

# 第二个子图
ax12 = fig.add_subplot(gs[0, 1])
ax12.bar(range(len(values2)), values2, color=colors, label=[None] * len(values2))
ax12.set_ylabel(y2_label)
ax12.set_xlabel('Evaluative Aspects')
ax12.set_xticks([])  # 隐藏横坐标标签
#ax12.legend(handles, labels, loc='best', frameon=False)


# 第三个子图
ax13 = fig.add_subplot(gs[0, 2])
ax13.bar(range(len(values3)), values3, color=colors, label=[None] * len(values3))
ax13.set_ylabel(y3_label)
ax13.set_xlabel('Evaluative Aspects')
ax13.set_xticks([])  # 隐藏横坐标标签
#ax13.legend(handles, labels, loc='best', frameon=False)

if SAVE_FIG:
    plt.savefig(save_figs_to + 'mse2.png', dpi=300)
    plt.savefig(save_figs_to + 'mse2.pdf', dpi=300)
    plt.savefig(save_figs_to + 'mse2.eps', dpi=300, format='eps')
    plt.savefig(save_figs_to + 'mse2.tif', dpi=300, format='tiff')
plt.show()