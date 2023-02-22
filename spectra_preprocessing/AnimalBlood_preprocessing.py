from turtle import st
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as SG
import os
import pathlib
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
from BaselineRemoval import BaselineRemoval


def spectra_preprocessing(filename):
    file_path = dir_path + '/' + file_dir + '/' + filename
    x, y = np.loadtxt(file_path, dtype=float, comments='#', delimiter=',', unpack=True)
    # 降噪
    denoising_y = signal.savgol_filter(y, 11, 3)        # 窗口长度，阶数
    baseObj = BaselineRemoval(denoising_y)              # 创建基线校正对象
    Zhangfit_output = baseObj.ZhangFit()                # airPLS
    for i in range(0, len(Zhangfit_output)):            # 负值处理
        if (Zhangfit_output[i] < 0):
            Zhangfit_output[i] = 0
    y = MinMaxScaler().fit_transform(Zhangfit_output.reshape(-1, 1)).reshape(1, -1)[0]    # 归一化

    # 降采样
    max_value, min_value = 10, -10  
    y_line = SG.LineString(list(zip(x, y)))
    des_file_dir = pathlib.Path(des_path + '/' + file_dir)
    if not os.path.exists(des_file_dir):
        os.mkdir(des_file_dir)
    with open(des_path + '/' + file_dir + '/' + filename, 'w') as f:
        for i in range(166, 2086, 2):       # 反射式间隔(200, 3000)  透射式间隔(166, 2086)
            x_line = SG.LineString([(i, min_value), (i, max_value)])
            cord = np.array(x_line.intersection(y_line))
            if cord.size == 0:
                f.write(str(i) + ', ' + str(0.) + '\n')
            else:
                f.write(str(i) + ', ' + str(cord[1]) + '\n')
    
# 反射式
# dir_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Reflective_animalBlood_removeString'
# des_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Reflective_animalBlood_preprocessing'
# 透射式
dir_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Transmissive_animalBlood_removeString'
des_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Transmissive_animalBlood_preprocessing'
i = 1
file_dir_list = os.listdir(dir_path)
for file_dir in file_dir_list:
    file_list = os.listdir(dir_path + '/' + file_dir)
    for filename in file_list:
        spectra_preprocessing(filename)
        print(i)
        i += 1
