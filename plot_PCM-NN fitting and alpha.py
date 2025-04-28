import sys

sys.path.insert(0, '../../Utilities/')

import os

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False  # 通常不需要设置为 True，除非你有特殊的 LaTeX 需求


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
data_frame20202023 = pandas.read_csv('Data\Data_2020-2023.csv')
alpha = np.loadtxt('Model0924\Train-Results-09-24-set16\Alpha1.txt')
X = np.loadtxt('Model0910\Train-Results-09-10-set2\X.txt')

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
ax11.plot(t_star, data_frame20202023.iloc[:, 1], lw=3,label='True data',color=(149/255, 163/255, 15/255), marker='o')
ax11.plot(t_star, X, lw=3,label='PCM-NN fitting',color=(22/255, 66/255, 38/255))
ax11.set_xlabel('Time')
ax11.set_ylabel('Number of Soybean Pod Borer Population')
ax11.legend(loc='best', frameon=False)
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.tick_params(axis='x', labelrotation=0)
ax11.text(0.5, -0.11, '(a)', transform=ax11.transAxes, fontsize=12,ha='center', va='top')

ax12 = fig.add_subplot(gs[0,1])
ax12.plot(t_star, alpha, lw=3, label='α(T,H,t)', color=(142/255, 15/255, 40/255))
ax12.set_xlabel('Time')
ax12.set_ylabel('The time-varying parameter of the model')
ax12.legend(loc='best', frameon=False)
ax12.spines['right'].set_visible(False)
ax12.spines['top'].set_visible(False)
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax12.tick_params(axis='x', labelrotation=0)
ax12.text(0.5, -0.11, '(b)', transform=ax12.transAxes, fontsize=12,ha='center', va='top')
# 添加红线
ax12.axhline(y=0, color='red',lw=2,  linestyle='--')
# 添加绿线
ax12.axhline(y=-0.372, color='black',lw=2,  linestyle='--')
x_pos = t_star[-6] - np.timedelta64(3, 'D')  # 稍微靠前一点，避免与最后一个数据点重叠
y_pos_positive = 0.03  # 根据你的数据范围调整这个值
ax12.text(x_pos, y_pos_positive, 'Growth Facilitation', transform=ax12.transData, fontsize=12, ha='left', va='bottom', color='red')

y_pos_negative = -0.03  # 根据你的数据范围调整这个值
ax12.text(x_pos, y_pos_negative, 'Growth Suppression', transform=ax12.transData, fontsize=12, ha='left', va='top', color='red')

x_pos1 = t_star[11] - np.timedelta64(3, 'D')  # 稍微靠前一点，避免与最后一个数据点重叠
# 添加绿线标签
y_pos_green_positive = -0.35  # 根据你的数据范围调整这个值
ax12.text(x_pos1, y_pos_green_positive, 'Positive Correlation', transform=ax12.transData, fontsize=12, ha='right',
          va='bottom', color='black')

y_pos_green_negative = -0.4  # 根据你的数据范围调整这个值
ax12.text(x_pos1, y_pos_green_negative, 'Negative Correlation', transform=ax12.transData, fontsize=12, ha='right',
          va='top', color='black')

ax12.fill_between(t_star, -1, 0, where=(t_star >= t_star[0]) & (t_star <= t_star[-1]), color='black', alpha=0.15, interpolate=True, step='pre')

ax12.fill_between(t_star, -1, -0.372, where=(t_star >= t_star[0]) & (t_star <= t_star[-1]), color='black', alpha=0.45, interpolate=True, step='pre')


output_base_name = 'PCM-NN fitting and alpha'
plt.savefig(f'{output_base_name}.pdf', format='pdf')
plt.savefig(f'{output_base_name}.png', format='png')


plt.show()