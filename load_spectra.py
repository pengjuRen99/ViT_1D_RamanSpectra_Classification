import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/renpengju/Documents/python_codes/experience_code/Data_set/Reflective_animalBlood_preprocessing/熊猫/1b928f30-7711-4ac6-9e6e-084b453b06f9.dx'
x, y = np.loadtxt(file_path, dtype=float, comments='#', delimiter=',', unpack=True)
plt.plot(x, y, color='black', label='Panda')
# plt.legend(loc=0, prop={'size':16})
plt.show()

