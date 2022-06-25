import os
import numpy as np
import pandas as pd


def load_spectrum_file(dir_path, des_path):
    file_dir_list = os.listdir(dir_path)
    for file_dir in file_dir_list:
        file_list = os.listdir(dir_path + '/' + file_dir)
        if not os.path.exists(des_path + '/' + file_dir):
            os.mkdir(des_path + '/' + file_dir)
        for filename in file_list:
            file_path = dir_path + '/' + file_dir + '/' + filename
            des_file_path = des_path + '/' + file_dir + '/' + filename
            print(file_path)
            with open(file_path, 'r') as f:
                data = f.read()
                data = data.translate(str.maketrans('', '', ';'))
                with open(des_file_path, 'w') as des_file:
                    des_file.write(data)
                    des_file.close()
                f.close()


dir_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Transmissive_animalBlood'
des_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Transmissive_animalBlood_removeString'
print('Loading data...')
load_spectrum_file(dir_path, des_path)
