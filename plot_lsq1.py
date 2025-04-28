import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置路径和加载数据
sys.path.insert(0, '../../Utilities/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_frame = pd.read_csv('Data/Data_2020-2024lsq.csv')

# 数据拟合时间长度
first_date = np.datetime64('2020-07-25')
last_date = np.datetime64('2020-08-15')  # 确保包含 8-15
t_star = np.arange(first_date, last_date + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))  # 修改这里，添加 +1 天

# 存储路径
current_directory = os.getcwd()
relative_path_figs = '/Data/'
save_figs_to = current_directory + relative_path_figs
SAVE_FIG = True

# 绘制图表
fig = plt.figure(figsize=(20, 6))
ax11 = fig.add_subplot(111)
bar_width = 0.8
ax11.bar(t_star, data_frame.iloc[:22, 1], width=bar_width, label='Annual average data',
         color=(149/255, 163/255, 15/255), align='center')
ax11.plot(t_star, data_frame.iloc[:22, 4], label='Nonlinear Least Square Fitting', lw=3, color=(22/255, 66/255, 38/255))
ax11.set_xlabel('Time')
ax11.set_ylabel('Number')
ax11.legend(loc='best', frameon=False)
ax11.spines['right'].set_visible(False)
ax11.spines['top'].set_visible(False)
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.tick_params(axis='x', labelrotation=0)

output_base_name = 'lsq'
plt.savefig(f'{output_base_name}.pdf', format='pdf')
plt.savefig(f'{output_base_name}.png', format='png')

plt.show()