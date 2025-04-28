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
# 加载原始数据
N = 5.5e6 + 13 + 306
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
# 数据拟合时间长度
first_date = np.datetime64('2023-03-26')
last_date = np.datetime64('2024-01-07') + np.timedelta64(7, 'D')
t_star = np.arange(first_date, last_date, np.timedelta64(7, 'D'))

data_frame_ture = pandas.read_csv('Data/BC_pre_data.csv')
I1_new_ture = data_frame_ture['I1new']  # T x 1 array
I2_new_ture = data_frame_ture['I2new']  # T x 1 array
I1_sum_ture = data_frame_ture['I1sum']  # T x 1 array
I2_sum_ture = data_frame_ture['I2sum']  # T x 1 array
I1_new_ture = I1_new_ture.to_numpy(dtype=np.float64)
I2_new_ture = I2_new_ture.to_numpy(dtype=np.float64)
I1_sum_ture = I1_sum_ture.to_numpy(dtype=np.float64)
I2_sum_ture = I2_sum_ture.to_numpy(dtype=np.float64)

#数据预测时间长度
first_date_pred = np.datetime64('2024-01-07')
last_date_pred = np.datetime64('2024-04-21') + np.timedelta64(7, 'D')
data_pred = np.arange(first_date_pred, last_date_pred, np.timedelta64(7, 'D'))


# 加载实验结果
sf=1e-4
dt_string =  '05-16'
I1_sum_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_sum_pred_mean.txt')
I1_sum_PINN = I1_sum_PINN/sf
I2_sum_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_sum_pred_mean.txt')
I2_sum_PINN = I2_sum_PINN/sf
I1_new_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_new_pred_mean.txt')
I1_new_PINN = I1_new_PINN/sf
I2_new_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_new_pred_mean.txt')
I2_new_PINN = I2_new_PINN/sf
S_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/S_pred_mean.txt')
S_PINN = S_PINN/sf
I1_new_ode_mean = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_new_ode_mean.txt')
I1_sum_ode_mean = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I1_sum_ode_mean.txt')
I2_new_ode_mean = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_new_ode_mean.txt')
I2_sum_ode_mean = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/I2_sum_ode_mean.txt')

# 参数结果
BetaI1_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/BetaI1_pred_mean.txt')
BetaI2_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/BetaI2_pred_mean.txt')
Gamma1_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/Gamma1_pred_mean.txt')
Gamma2_PINN = np.loadtxt('Model1/Train-Results-'+dt_string+'-Average/Gamma2_pred_mean.txt')



# 预测结果
## I1 new  and sum
newI1_mean= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI1_mean.txt')
newI1_lb_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI1_lb_d0.txt')
newI1_ub_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI1_ub_d0.txt')
newI1_lb_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI1_lb_d1.txt')
newI1_ub_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI1_ub_d1.txt')
sumI1_mean= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI1_mean.txt')
sumI1_lb_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI1_lb_d0.txt')
sumI1_ub_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI1_ub_d0.txt')
sumI1_lb_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI1_lb_d1.txt')
sumI1_ub_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI1_ub_d1.txt')


## I2 new  and sum
newI2_mean= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI2_mean.txt')
newI2_lb_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI2_lb_d0.txt')
newI2_ub_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI2_ub_d0.txt')
newI2_lb_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI2_lb_d1.txt')
newI2_ub_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/newI2_ub_d1.txt')
sumI2_mean= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI2_mean.txt')
sumI2_lb_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI2_lb_d0.txt')
sumI2_ub_d0= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI2_ub_d0.txt')
sumI2_lb_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI2_lb_d1.txt')
sumI2_ub_d1= np.loadtxt('Model1/Prediction-Results-'+dt_string+'/sumI2_ub_d1.txt')


                        ######################################################################
                        ############################# Plotting ###############################
                        ######################################################################
# 存储路径
current_directory = os.getcwd()
relative_path_figs = '/figs_con/'
save_figs_to = current_directory + relative_path_figs
SAVE_FIG = True


# ########################   I1+I2 ########################
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
# bar_width = 0.35
# index = np.arange(len(t_star))
#
#
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# #自定义颜色
# color1 = (237/255, 173/255, 197/255)  # I1_new
# color2 = (108/255, 190/255, 195/255)  # I1_sum
# ax1.set_position([0.1, 0.1, 0.35, 0.8])  # [left, bottom, width, height]
# # 创建条形图
# bars1 = ax1.bar(index, I1_new_star, bar_width, label='New reported cases', color=color1)
# bars2 = ax1.bar(index + bar_width, I1_sum_star, bar_width, label='Cumulative reported cases', color=color2)
# ax1.set_title('BC')
# ax1.set_xlabel('Times')
# ax1.set_ylabel('Weekly reported cases of XBB.$1.16.^{*}$')
# ax1.set_xticks(index[::8] + bar_width / 2)
# ax1.set_xticklabels(t_star[::8])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax1.legend(frameon=False)
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(122)
# #自定义颜色
# color1 = (206 / 255, 170 / 255, 208 / 255)  # new-case
# color2 = (97 / 255, 156 / 255, 217 / 255)  # sum-case
# ax2.set_position([0.55, 0.1, 0.35, 0.8])
# # 创建条形图
# bars1 = ax2.bar(index, I2_new_star, bar_width, label='New reported cases', color=color1)
# bars2 = ax2.bar(index + bar_width, I2_sum_star, bar_width, label='Cumulative reported cases', color=color2)
# ax2.set_title('BC')
# ax2.set_xlabel('Times')
# ax2.set_ylabel('Weekly reported cases of XBB.$1.5.^{*}$')
# ax2.set_xticks(index[::8] + bar_width / 2)
# ax2.set_xticklabels(t_star[::8])
# ax2.legend(frameon=False)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.tight_layout(w_pad=4)
#
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'I_data.png', dpi=300)
#     plt.savefig(save_figs_to + 'I_data.pdf', dpi=300)
# plt.show()



### 不要
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
# bar_width = 0.35
# index = np.arange(len(t_star))
#
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# ax1.set_position([0.1, 0.1, 0.35, 0.8])  # [left, bottom, width, height]
# # 创建条形图
# bars1 = ax1.bar(index, I1_new_star, bar_width, label='New reported cases', color=(237/255, 173/255, 197/255))
# bars2 = ax1.bar(index + bar_width, I1_sum_star, bar_width, label='Cumulative reported cases', color=(108/255, 190/255, 195/255))
# # 添加 I1-pinn
# ax1.plot(index[1:], I1_new_PINN, label='New reported cases--PINN', color=(149/255, 132/255, 193/255),linewidth=2,linestyle='-')
# ax1.plot(index + bar_width, I1_sum_PINN, label='Cumulative reported cases--PINN', color=(97/255, 156/255, 217/255),linewidth=2,linestyle='--')
# ax1.set_title('BC')
# ax1.set_xlabel('Times')
# ax1.set_ylabel('Weekly reported cases of XBB.$1.16^{*}$')
# ax1.set_xticks(index[::8] + bar_width / 2)
# ax1.set_xticklabels(t_star[::8])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
# ax3 = ax1.twinx()
# ax3.plot(index + bar_width / 2, BetaI1_PINN, label=r'$\beta_{1}(t)$--PINN', color=(217/255, 66/255, 60/255),  linewidth=2, linestyle=':')
# ax3.plot(index + bar_width / 2, Gamma1_PINN, label=r'$\gamma_{1}(t)$--PINN', color=(71/255, 88/255, 162/255) ,  linewidth=2,linestyle='-.')
# ax3.set_ylabel('The time-varying parameters of the model')
# ax3.set_yticks(np.linspace(0, 1, 11))
# ax3.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax3.get_legend_handles_labels()
# ax3.legend(lines + lines2, labels + labels2, frameon=False, fontsize='large', loc='upper center')
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(122)
# ax2.set_position([0.56, 0.1, 0.35, 0.8])
# # 创建条形图
# bars1 = ax2.bar(index, I2_new_star, bar_width, label='New reported cases', color=(206 / 255, 170 / 255, 208 / 255))
# bars2 = ax2.bar(index + bar_width, I2_sum_star, bar_width, label='Cumulative reported cases', color=(97 / 255, 156 / 255, 217 / 255) )
# # 添加 I2-pinn
# ax2.plot(index[1:], I2_new_PINN, label='New reported cases--PINN', color=(149/255, 132/255, 193/255) ,linewidth=2,linestyle='-')
# ax2.plot(index + bar_width, I2_sum_PINN, label='Cumulative reported cases--PINN', color=(97/255, 156/255, 217/255),linewidth=2,linestyle='--')
# ax2.set_title('BC')
# ax2.set_xlabel('Times')
# ax2.set_ylabel('Weekly reported cases of XBB.1.5')
# ax2.set_xticks(index[::8] + bar_width / 2)
# ax2.set_xticklabels(t_star[::8])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#
# ax4 = ax2.twinx()
# ax4.plot(index + bar_width / 2, BetaI2_PINN, label=r'$\beta_{2}(t)$--PINN', color=(217/255, 66/255, 60/255)  ,  linewidth=2, linestyle=':')
# ax4.plot(index + bar_width / 2, Gamma2_PINN, label=r'$\gamma_{2}(t)$--PINN', color=(71/255, 88/255, 162/255) ,  linewidth=2,linestyle='-.')
# ax4.set_ylabel('The time-varying parameters of the model')
# ax4.set_yticks(np.linspace(0, 1, 11))
# ax4.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
# lines, labels = ax2.get_legend_handles_labels()
# lines2, labels2 = ax4.get_legend_handles_labels()
# ax4.legend(lines + lines2, labels + labels2, frameon=False, fontsize='large', loc='upper center')
# plt.tight_layout(w_pad=3)
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'I_fit.png', dpi=300)
#     plt.savefig(save_figs_to + 'I_fit.pdf', dpi=300)
# plt.show()

#
###  竖放  要
# 设置大图的尺寸
fig = plt.figure(figsize=(14, 10))  # 大图尺寸：宽14英寸，高6英寸
bar_width = 0.35
index = np.arange(len(t_star))

# 添加第一个子图，并设置其位置和尺寸
ax1 = fig.add_subplot(211)
ax1.set_position([0.1, 0.55, 0.8, 0.2])  # [left, bottom, width, height]
# 创建条形图
bars1 = ax1.bar(index, I1_new_star, bar_width, label='New reported cases', color=(237/255, 173/255, 197/255))
bars2 = ax1.bar(index + bar_width, I1_sum_star, bar_width, label='Cumulative reported cases', color=(108/255, 190/255, 195/255))
# 添加 I1-pinn
ax1.plot(index[1:], I1_new_PINN, label='New reported cases--IDINN', color=(149/255, 132/255, 193/255),linewidth=2,linestyle='-')
ax1.plot(index + bar_width, I1_sum_PINN, label='Cumulative reported cases--IDINN', color=(97/255, 156/255, 217/255),linewidth=2,linestyle='--')
ax1.set_title('BC')
ax1.set_xlabel('Times')
ax1.set_ylabel('Weekly reported cases of XBB.$1.16.^{*}$')
ax1.set_xticks(index[::8] + bar_width / 2)
ax1.set_xticklabels(t_star[::8])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax3 = ax1.twinx()
ax3.plot(index + bar_width / 2, BetaI1_PINN, label=r'$\beta_{1}(t)$--IDINN', color=(217/255, 66/255, 60/255),  linewidth=2, linestyle=':')
ax3.plot(index + bar_width / 2, Gamma1_PINN, label=r'$\gamma_{1}(t)$--IDINN', color=(71/255, 88/255, 162/255) ,  linewidth=2,linestyle='-.')
ax3.set_ylabel('The time-varying parameters of the model')
ax3.set_yticks(np.linspace(0, 1, 11))
ax3.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax3.legend(lines + lines2, labels + labels2, frameon=False,  loc='upper left')

# 添加第二个子图，并设置其位置和尺寸
ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.1, 0.8, 0.35])
# 创建条形图
bars1 = ax2.bar(index, I2_new_star, bar_width, label='New reported cases', color=(206 / 255, 170 / 255, 208 / 255))
bars2 = ax2.bar(index + bar_width, I2_sum_star, bar_width, label='Cumulative reported cases', color=(97 / 255, 156 / 255, 217 / 255) )
# 添加 I2-pinn
ax2.plot(index[1:], I2_new_PINN, label='New reported cases--IDINN', color=(149/255, 132/255, 193/255) ,linewidth=2,linestyle='-')
ax2.plot(index + bar_width, I2_sum_PINN, label='Cumulative reported cases--IDINN', color=(97/255, 156/255, 217/255),linewidth=2,linestyle='--')
ax2.set_title('BC')
ax2.set_xlabel('Times')
ax2.set_ylabel('Weekly reported cases of XBB.$1.5.^{*}$')
ax2.set_xticks(index[::8] + bar_width / 2)
ax2.set_xticklabels(t_star[::8])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax4 = ax2.twinx()
ax4.plot(index + bar_width / 2, BetaI2_PINN, label=r'$\beta_{2}(t)$--IDINN', color=(217/255, 66/255, 60/255)  ,  linewidth=2, linestyle=':')
ax4.plot(index + bar_width / 2, Gamma2_PINN, label=r'$\gamma_{2}(t)$--IDINN', color=(71/255, 88/255, 162/255) ,  linewidth=2,linestyle='-.')
ax4.set_ylabel('The time-varying parameters of the model')
ax4.set_yticks(np.linspace(0, 1, 11))
ax4.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax4.legend(lines + lines2, labels + labels2, frameon=False, loc='upper left')
plt.tight_layout(h_pad=2)
# 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
fig.set_size_inches(11, 10)
if SAVE_FIG:
    plt.savefig(save_figs_to + 'I_fit.png', dpi=300)
    plt.savefig(save_figs_to + 'I_fit.pdf', dpi=300)
plt.show()


# #################### R0 曲线  ################
# R_O=BetaI1_PINN/(Gamma1_PINN+0.000023)
# R_O1=R_O*((S_PINN)/N)
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
#
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# ax1.set_position([0.1, 0.1, 0.35, 0.8])  # [left, bottom, width, height]
# # 创建条形图
# ax1.plot(t_star, R_O, 'k-', lw=1, label='$\mathcal{R}_{0,1}(t)$',color=(217/255, 66/255, 60/255) ,linestyle='-.')
# ax1.plot(t_star, R_O1, 'k-', lw=1, label='$\mathcal{R}_{e,1}(t)$',color=(133/255, 76/255, 152/255),linestyle='--')
# # 设置左侧 Y 轴标签为
# ax1.set_ylabel('Values')
# ax1.set_xlabel('Times')
# ax1.legend(frameon=False)
# ax1.set_xticks(t_star[::8])
# ax1.set_xticklabels(t_star[::8])
# # 创建嵌套的放大图区域
# left, bottom, width, height = [0.588, 0.65, 0.2, 0.2]
# ax_inset1 = ax1.inset_axes([left, bottom, width, height])
# ax_inset1.plot(t_star, R_O,lw=1,color=(217/255, 66/255, 60/255),linestyle='-.')
# ax_inset1.plot(t_star, R_O1,lw=1,color=(133/255, 76/255, 152/255),linestyle='--')
# ax_inset1.set_xlim(t_star[20], t_star[22])
# ax_inset1.set_ylim(1.3, 1.4)
# ax_inset1.set_xticks([t_star[20], t_star[22]])
# ax_inset1.set_xticklabels([t_star[20], t_star[22]])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# R_O=BetaI2_PINN/(Gamma2_PINN+0.000023)
# R_O1=R_O*((S_PINN)/N)
# ax2 = fig.add_subplot(122)
# ax2.set_position([0.56, 0.1, 0.35, 0.8])
# # 创建条形图
# ax2.plot(t_star, R_O, 'k-', lw=1, label='$\mathcal{R}_{0,2}(t)$',color=(217/255, 66/255, 60/255),linestyle='-.')
# ax2.plot(t_star, R_O1, 'k-', lw=1, label='$\mathcal{R}_{e,2}(t)$',color=(133/255, 76/255, 152/255) ,linestyle='--')
# ax2.set_ylabel('Values')
# ax2.set_xlabel('Times')
# ax2.legend(frameon=False, fontsize='large')
#
# # 创建嵌套的放大图区域
# left, bottom, width, height = [0.588, 0.65, 0.2, 0.2]
# ax_inset2 = ax2.inset_axes([left, bottom, width, height])
# ax_inset2.plot(t_star, R_O,lw=1,color=(217/255, 66/255, 60/255),linestyle='-.')
# ax_inset2.plot(t_star, R_O1,lw=1,color=(133/255, 76/255, 152/255),linestyle='--')
# ax_inset2.set_xlim(t_star[23], t_star[25])
# ax_inset2.set_ylim(1.2, 1.3)
# ax_inset2.set_xticks([t_star[23], t_star[25]])
# ax_inset2.set_xticklabels([t_star[23], t_star[25]])
# ax2.set_xticks(t_star[::8])
# ax2.set_xticklabels(t_star[::8])
# plt.tight_layout(w_pad=4)
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'R0.png', dpi=300)
#     plt.savefig(save_figs_to + 'R0.pdf', dpi=300)
# plt.show()

# ###  竖放 要
# #################### R0 曲线  ################
# R_O=BetaI1_PINN/(Gamma1_PINN+0.000023)
# R_O1=R_O*((S_PINN)/N)
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 11))  # 大图尺寸：宽14英寸，高6英寸
#
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 创建条形图
# ax1.plot(t_star, R_O, 'k-', lw=1, label='$\mathcal{R}_{0,1}(t)$',color=(217/255, 66/255, 60/255) ,linestyle='-.')
# ax1.plot(t_star, R_O1, 'k-', lw=1, label='$\mathcal{R}_{e,1}(t)$',color=(133/255, 76/255, 152/255),linestyle='--')
# # 设置左侧 Y 轴标签为
# ax1.set_ylabel('Values')
# ax1.set_xlabel('Times')
# ax1.legend(frameon=False)
# ax1.set_xticks(t_star[::8])
# ax1.set_xticklabels(t_star[::8])
# # 创建嵌套的放大图区域
# left, bottom, width, height = [0.7, 0.65, 0.2, 0.2]
# ax_inset1 = ax1.inset_axes([left, bottom, width, height])
# ax_inset1.plot(t_star, R_O,lw=1,color=(217/255, 66/255, 60/255),linestyle='-.')
# ax_inset1.plot(t_star, R_O1,lw=1,color=(133/255, 76/255, 152/255),linestyle='--')
# ax_inset1.set_xlim(t_star[20], t_star[22])
# ax_inset1.set_ylim(1.3, 1.4)
# ax_inset1.set_xticks([t_star[20], t_star[22]])
# ax_inset1.set_xticklabels([t_star[20], t_star[22]])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# R_O=BetaI2_PINN/(Gamma2_PINN+0.000023)
# R_O1=R_O*((S_PINN)/N)
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# # 创建条形图
# ax2.plot(t_star, R_O, 'k-', lw=1, label='$\mathcal{R}_{0,2}(t)$',color=(217/255, 66/255, 60/255),linestyle='-.')
# ax2.plot(t_star, R_O1, 'k-', lw=1, label='$\mathcal{R}_{e,2}(t)$',color=(133/255, 76/255, 152/255) ,linestyle='--')
# ax2.set_ylabel('Values')
# ax2.set_xlabel('Times')
# ax2.legend(frameon=False, fontsize='large')
#
# # 创建嵌套的放大图区域
# left, bottom, width, height = [0.67, 0.65, 0.2, 0.2]
# ax_inset2 = ax2.inset_axes([left, bottom, width, height])
# ax_inset2.plot(t_star, R_O,lw=1,color=(217/255, 66/255, 60/255),linestyle='-.')
# ax_inset2.plot(t_star, R_O1,lw=1,color=(133/255, 76/255, 152/255),linestyle='--')
# ax_inset2.set_xlim(t_star[23], t_star[25])
# ax_inset2.set_ylim(1.2, 1.3)
# ax_inset2.set_xticks([t_star[23], t_star[25]])
# ax_inset2.set_xticklabels([t_star[23], t_star[25]])
# ax2.set_xticks(t_star[::8])
# ax2.set_xticklabels(t_star[::8])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'R0.png', dpi=300)
#     plt.savefig(save_figs_to + 'R0.pdf', dpi=300)
# plt.show()



# ############ beta
# #BetaI1 curve
# beta_pred_0 = np.array([BetaI1_PINN[-1] for i in range(data_pred.shape[0])])
# gamma_pred_0 = np.array([Gamma1_PINN[-1] for i in range(data_pred.shape[0])])
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# ax1.plot(t_star, BetaI1_PINN,  lw=2, color=(97/255, 156/255, 217/255) , linestyle='-.',label=r'$\beta_{1}(t)$--PINN')
# ax1.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255), label='Prediction')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.1), \
#                  beta_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.2), \
#                  beta_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # gamma 1
# ax1.plot(t_star, Gamma1_PINN,  lw=2, color=(237/255, 173/255, 197/255), linestyle='--',label=r'$\gamma_{1}(t)$--PINN')
# ax1.plot(data_pred.flatten(), gamma_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.1), \
#                  gamma_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True)
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.2), \
#                  gamma_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True)
#
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
# ax1.set_ylabel('The time-varying parameters of the model')
# ax1.set_xlabel('Times')
#
#
# # 添加第二个子图，并设置其位置和尺寸
# beta_pred_0 = np.array([BetaI2_PINN[-1] for i in range(data_pred.shape[0])])
# gamma_pred_0 = np.array([Gamma2_PINN[-1] for i in range(data_pred.shape[0])])
# ax2 = fig.add_subplot(122)
# ax2.plot(t_star, BetaI2_PINN,  lw=2, color=(97/255, 156/255, 217/255) , linestyle='-.',label=r'$\beta_{2}(t)$--PINN')
# ax2.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255), label='Prediction')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.1), \
#                  beta_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.2), \
#                  beta_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # gamma 1
# ax2.plot(t_star, Gamma2_PINN,  lw=2, color=(237/255, 173/255, 197/255), linestyle='--',label=r'$\gamma_{2}(t)$--PINN')
# ax2.plot(data_pred.flatten(), gamma_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.1), \
#                  gamma_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True)
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.2), \
#                  gamma_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True)
#
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('The time-varying parameters of the model')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# plt.tight_layout(w_pad=2)
#
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'par.png', dpi=300)
#     plt.savefig(save_figs_to + 'par.pdf', dpi=300)
# plt.show()


#  #######  竖放  要
# #################### par 曲线  ################
# #BetaI1 curve
# beta_pred_0 = np.array([BetaI1_PINN[-1] for i in range(data_pred.shape[0])])
# gamma_pred_0 = np.array([Gamma1_PINN[-1] for i in range(data_pred.shape[0])])
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 11))  # 大图尺寸：宽14英寸，高6英寸
#
# # 添加第一个子图，并设置其位置和尺寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 添加第一个子图，并设置其位置和尺寸
# ax1.plot(t_star, BetaI1_PINN,  lw=2, color=(97/255, 156/255, 217/255) , linestyle='-.',label=r'$\beta_{1}(t)$--IDINN')
# ax1.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.1), \
#                  beta_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.2), \
#                  beta_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # gamma 1
# ax1.plot(t_star, Gamma1_PINN,  lw=2, color=(237/255, 173/255, 197/255), linestyle='--',label=r'$\gamma_{1}(t)$--IDINN')
# ax1.plot(data_pred.flatten(), gamma_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255),label='Prediction')
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.1), \
#                  gamma_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True)
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.2), \
#                  gamma_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True)
#
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
# ax1.set_ylabel('Time-varying parameters')
# ax1.set_xlabel('Times')
#
#
# # 添加第二个子图，并设置其位置和尺寸
# beta_pred_0 = np.array([BetaI2_PINN[-1] for i in range(data_pred.shape[0])])
# gamma_pred_0 = np.array([Gamma2_PINN[-1] for i in range(data_pred.shape[0])])
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# ax2.plot(t_star, BetaI2_PINN,  lw=2, color=(97/255, 156/255, 217/255) , linestyle='-.',label=r'$\beta_{2}(t)$--IDINN')
# ax2.plot(data_pred.flatten(), beta_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.1), \
#                  beta_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                  beta_pred_0*(1.2), \
#                  beta_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # gamma 2
# ax2.plot(t_star, Gamma2_PINN,  lw=2, color=(237/255, 173/255, 197/255), linestyle='--',label=r'$\gamma_{2}(t)$--IDINN')
# ax2.plot(data_pred.flatten(), gamma_pred_0, 'm--', lw=2, color=(147/255, 85/255, 176/255), label='Prediction')
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.1), \
#                  gamma_pred_0*(0.9), \
#                  facecolor=(108/255,190/255,195/255,1), interpolate=True)
# plt.fill_between(data_pred.flatten(), \
#                  gamma_pred_0*(1.2), \
#                  gamma_pred_0*(0.8), \
#                  facecolor=(170/255,215/255,200/255,0.6), interpolate=True)
#
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('Time-varying parameters')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'par.png', dpi=300)
#     plt.savefig(save_figs_to + 'par.pdf', dpi=300)
# plt.show()


# # ############ I1 +I2 new预测数据
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# ax1.plot(t_star, I1_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(data_pred[1:-1], I1_new_ture[:-1], ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax1.plot(t_star[1:], I1_new_PINN,  lw=2, label='PINN-Training', color=(97/255, 156/255, 217/255))
# ax1.plot(data_pred[:-1], newI1_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI1_lb_d0.flatten(), newI1_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI1_lb_d1.flatten(), newI1_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_ylabel('Weekly reported  cases  ($I_{1}^{n}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(122)
# ax2.plot(t_star, I2_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(data_pred[1:-1], I2_new_ture[:-1], ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax2.plot(t_star[1:], I2_new_PINN,  lw=2, label='PINN-Training', color=(97/255, 156/255, 217/255))
# ax2.plot(data_pred[:-1], newI2_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI2_lb_d0.flatten(), newI2_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI2_lb_d1.flatten(), newI2_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('Weekly reported  cases  ($I_{2}^{n}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# plt.tight_layout(w_pad=2)
#
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'new_case_pre.png', dpi=300)
#     plt.savefig(save_figs_to + 'new_case_pre.pdf', dpi=300)
# plt.show()





# ########   竖放 要
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 11))  # 大图尺寸：宽14英寸，高6英寸
# # 添加第一个子图，并设置其位置和尺寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 添加第一个子图，并设置其位置和尺寸
# ax1.plot(t_star, I1_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(data_pred[1:-1], I1_new_ture[:-1], ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax1.plot(t_star[1:], I1_new_PINN,  lw=2, label='IDINN-Training', color=(97/255, 156/255, 217/255))
# ax1.plot(data_pred[:-1], newI1_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI1_lb_d0.flatten(), newI1_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI1_lb_d1.flatten(), newI1_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_ylabel('Weekly reported  cases  ($I_{1}^{n}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# ax2.plot(t_star, I2_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(data_pred[1:-1], I2_new_ture[:-1], ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax2.plot(t_star[1:], I2_new_PINN,  lw=2, label='IDINN-Training', color=(97/255, 156/255, 217/255))
# ax2.plot(data_pred[:-1], newI2_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI2_lb_d0.flatten(), newI2_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred[:-1].flatten(), \
#                   newI2_lb_d1.flatten(), newI2_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('Weekly reported  cases  ($I_{2}^{n}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'new_case_pre.png', dpi=300)
#     plt.savefig(save_figs_to + 'new_case_pre.pdf', dpi=300)
# plt.show()



# # ############ I1 +I2 sum 预测数据
# # 设置大图的尺寸
# fig = plt.figure(figsize=(14, 6))  # 大图尺寸：宽14英寸，高6英寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# # 添加第一个子图，并设置其位置和尺寸
# ax1 = fig.add_subplot(121)
# ax1.plot(t_star, I1_sum_star,   '--',lw=3, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(data_pred[1:], I1_sum_ture, ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax1.plot(t_star, I1_sum_PINN,  lw=1, label='PINN-Training', color=(97/255, 156/255, 217/255))
# ax1.plot(data_pred, sumI1_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                   sumI1_lb_d0.flatten(), sumI1_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                   sumI1_lb_d1.flatten(), sumI1_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_ylabel('Cumulative reported  cases  ($I_{1}^{c}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(122)
# ax2.plot(t_star, I2_sum_star,   '--',lw=3, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(data_pred[1:], I2_sum_ture, ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax2.plot(t_star, I2_sum_PINN,  lw=1, label='PINN-Training', color=(97/255, 156/255, 217/255))
# ax2.plot(data_pred, sumI2_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                   sumI2_lb_d0.flatten(), sumI2_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                   sumI2_lb_d1.flatten(), sumI2_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('Cumulative reported  cases  ($I_{2}^{c}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# plt.tight_layout(w_pad=2)
#
# # 调整子图大小
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
# fig.set_size_inches(14, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'sum_case_pre.png', dpi=300)
#     plt.savefig(save_figs_to + 'sum_case_pre.pdf', dpi=300)
# plt.show()


# ########   竖放 要
# # I1 +I2 sum
# # 设置大图的尺寸
# fig = plt.figure(figsize=(8, 11))  # 大图尺寸：宽14英寸，高6英寸
# # 添加第一个子图，并设置其位置和尺寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 添加第一个子图，并设置其位置和尺寸
# ax1.plot(t_star, I1_sum_star,   '--',lw=3, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(data_pred[1:], I1_sum_ture, ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax1.plot(t_star, I1_sum_PINN,  lw=1, label='IDINN-Training', color=(97/255, 156/255, 217/255))
# ax1.plot(data_pred, sumI1_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                   sumI1_lb_d0.flatten(), sumI1_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                   sumI1_lb_d1.flatten(), sumI1_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax1.legend(frameon=False)
# ax1.set_ylabel('Cumulative reported  cases  ($I_{1}^{c}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_all[::10])
# ax1.set_xticklabels(t_all[::10])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# ax2.plot(t_star, I2_sum_star,   '--',lw=3, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(data_pred[1:], I2_sum_ture, ':', lw=2, label='Testing Data',color=(149/255, 132/255, 193/255))
# ax2.plot(t_star, I2_sum_PINN,  lw=1, label='IDINN-Training', color=(97/255, 156/255, 217/255))
# ax2.plot(data_pred, sumI2_mean, '-.', lw=2, label='Prediction',color=(147/255, 85/255, 176/255))
# plt.fill_between(data_pred.flatten(), \
#                   sumI2_lb_d0.flatten(), sumI2_ub_d0.flatten(), \
#                   facecolor=(108/255,190/255,195/255,1), interpolate=True, label='Prediction-std-(10%)')
# plt.fill_between(data_pred.flatten(), \
#                   sumI2_lb_d1.flatten(), sumI2_ub_d1.flatten(), \
#                   facecolor=(170/255,215/255,200/255,0.6), interpolate=True, label='Prediction-std-(20%)')
# # 添加垂直线
# plt.axvline(x=t_star[-1], color=(239/255,127/255,87/255), lw=2, linestyle=':')
# ax2.legend(frameon=False)
# ax2.set_ylabel('Cumulative reported  cases  ($I_{2}^{c}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_all[::10])
# ax2.set_xticklabels(t_all[::10])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'sum_case_pre.png', dpi=300)
#     plt.savefig(save_figs_to + 'sum_case_pre.pdf', dpi=300)
# plt.show()




# ########   竖放 要
# # I1 +I2 new  ode
# # 设置大图的尺寸
# fig = plt.figure(figsize=(8, 11))  # 大图尺寸：宽14英寸，高6英寸
# # 添加第一个子图，并设置其位置和尺寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 添加第一个子图，并设置其位置和尺寸
# ax1.plot(t_star, I1_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(t_star[1:], I1_new_ode_mean, lw=2, label='ODE', color=(97/255, 156/255, 217/255))
# ax1.legend(frameon=False)
# ax1.set_ylabel('New reported  cases  ($I_{1}^{n}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_star[::8])
# ax1.set_xticklabels(t_star[::8])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# ax2.plot(t_star, I2_new_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(t_star[1:], I2_new_ode_mean, lw=2, label='ODE', color=(97/255, 156/255, 217/255))
# ax2.legend(frameon=False)
# ax2.set_ylabel('New reported  cases  ($I_{2}^{n}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_star[::8])
# ax2.set_xticklabels(t_star[::8])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'new_case_ode.png', dpi=300)
#     plt.savefig(save_figs_to + 'new_case_ode.pdf', dpi=300)
# plt.show()




# ########   竖放 要
# # I1 +I2 sum  ode
# # 设置大图的尺寸
# fig = plt.figure(figsize=(8, 11))  # 大图尺寸：宽14英寸，高6英寸
# # 添加第一个子图，并设置其位置和尺寸
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(211)
# ax1.set_position([0.1, 0.5, 0.8, 0.4])  # [left, bottom, width, height]
# # 添加第一个子图，并设置其位置和尺寸
# ax1.plot(t_star, I1_sum_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax1.plot(t_star, I1_sum_ode_mean, lw=2, label='ODE', color=(97/255, 156/255, 217/255))
# ax1.legend(frameon=False)
# ax1.set_ylabel('Cumulative reported  cases  ($I_{1}^{c}$)')
# ax1.set_xlabel('Times')
# ax1.set_xticks(t_star[::8])
# ax1.set_xticklabels(t_star[::8])
#
#
# # 添加第二个子图，并设置其位置和尺寸
# ax2 = fig.add_subplot(212)
# ax2.set_position([0.1, 0.08, 0.8, 0.4])
# ax2.plot(t_star, I2_sum_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(t_star, I2_sum_ode_mean, lw=2, label='ODE', color=(97/255, 156/255, 217/255))
# ax2.legend(frameon=False)
# ax2.set_ylabel('Cumulative reported  cases  ($I_{2}^{c}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_star[::8])
# ax2.set_xticklabels(t_star[::8])
# # 调整子图大小
# plt.tight_layout(h_pad=0.3)
# # plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.07)
# fig.set_size_inches(8, 6)
# if SAVE_FIG:
#     plt.savefig(save_figs_to + 'sum_case_ode.png', dpi=300)
#     plt.savefig(save_figs_to + 'sum_case_ode.pdf', dpi=300)
# plt.show()


#
# # 创建一个大图
# fig = plt.figure(figsize=(15, 10))
# bar_width = 0.35
# index = np.arange(len(t_star))
# # 创建一个GridSpec对象，指定总共2行3列
# gs = gridspec.GridSpec(2, 4, wspace=0.5, hspace=0.2)
#
# # 绘制不同大小的子图
# t_all = np.concatenate((t_star,data_pred[1:]), axis=0)
# ax1 = fig.add_subplot(gs[0, 0:2])
#
# color1 = (237/255, 173/255, 197/255)  # I1_new
# color2 = (108/255, 190/255, 195/255)  # I1_sum
# # ax1.set_position([0.1, 0.1, 0.35, 0.8])  # [left, bottom, width, height]
# # 创建条形图
# bars1 = ax1.bar(index, I1_new_star, bar_width, label='New reported cases', color=color1)
# bars2 = ax1.bar(index + bar_width, I1_sum_star, bar_width, label='Cumulative reported cases', color=color2)
# ax1.set_title('BC')
# ax1.set_xlabel('Times')
# ax1.set_ylabel('Weekly reported cases of XBB.$1.16^{*}$')
# ax1.set_xticks(index[::8] + bar_width / 2)
# ax1.set_xticklabels(t_star[::8])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax1.legend(frameon=False)
#
#
# ax2 = fig.add_subplot(gs[1, 0:])
#
# # 添加第一个子图，并设置其位置和尺寸
# ax2.plot(t_star, I1_sum_star,  '--',lw=2, label='Training Data', color=(237/255, 173/255, 197/255))
# ax2.plot(t_star, I1_sum_ode_mean, lw=2, label='ODE', color=(97/255, 156/255, 217/255))
# ax2.legend(frameon=False)
# ax2.set_ylabel('Cumulative reported  cases  ($I_{1}^{c}$)')
# ax2.set_xlabel('Times')
# ax2.set_xticks(t_star[::8])
# ax2.set_xticklabels(t_star[::8])
#
# # 显示图形
# plt.show()