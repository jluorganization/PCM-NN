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
#加载数据
data_frame20202024 = pandas.read_csv('Data\Data_2020-2024.csv')
data_frame20202023 = pandas.read_csv('Data\Data_2020-2023.csv')
data_frame2024 = pandas.read_csv('Data\Data_2024.csv')
X = np.loadtxt('Model0924\Train-Results-09-24-set16\X1_ode.txt')
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

####################################################### Data ##########################################################
# 绘制图表
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1,2, wspace=0.2, hspace=0.52)
ax11 = fig.add_subplot(gs[0, 0])
ax11.plot(t_star, data_frame20202023.iloc[:, 1], lw=3,label='True Data',color=(149/255, 163/255, 15/255), marker='o')
ax11.plot(t_star, X, lw=3,label='ODE Solver',color=(22/255, 66/255, 38/255))
ax11.set_xlabel('Time')
ax11.set_ylabel('Number of Soybean Pod Borer Population')
ax11.legend(loc='best', frameon=False)
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.tick_params(axis='x', labelrotation=0)
ax11.text(0.5,  -0.11, '(a)', transform=ax11.transAxes, fontsize=12,ha='center', va='top')

ax12 = fig.add_subplot(gs[0,1])
ax12.plot(t_star, data_frame20202024.iloc[:, 1], lw=3, label='2020-2024 Annual Average Data', color=(173/255, 45/255, 35/255),linestyle='--',alpha=0.4)

ax12.fill_between(t_star, data_frame20202024.iloc[:, 1],
                  y2=min(data_frame20202024.iloc[:, 1]) - 1,  # 假设一个下界，你可以根据需要调整
                  where=(data_frame20202024.iloc[:, 1] >= min(data_frame20202024.iloc[:, 1]) - 1),  # 确保填充在合理范围内
                  color=(173/255, 45/255, 35/255), alpha=0.3)  # 使用30%透明度的红色

ax12.plot(t_star, X24, lw=3, label='Prediction', color=(59/255, 5/255, 15/255))
ax12.set_xlabel('Time')
ax12.set_ylabel('Number of Soybean Pod Borer Population')
ax12.legend(loc='best', frameon=False)
ax12.spines['right'].set_visible(False)
ax12.spines['top'].set_visible(False)
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax12.tick_params(axis='x', labelrotation=0)
ax12.text(0.5,  -0.11, '(b)', transform=ax12.transAxes, fontsize=12,ha='center', va='top')

output_base_name = 'ode and prediction2'
plt.savefig(f'{output_base_name}.pdf', format='pdf')
plt.savefig(f'{output_base_name}.png', format='png')

plt.show()