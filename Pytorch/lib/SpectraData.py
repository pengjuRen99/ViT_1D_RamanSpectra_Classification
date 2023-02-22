from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pathlib
import torch
import numpy as np
import joblib


class SpectraData(Dataset):
    def __init__(self, spectra_path, mode='train'):
        self.spectra_path = spectra_path
        self.le = LabelEncoder()
        
        if mode == 'train':
            self.train_path = pathlib.Path(self.spectra_path + '/train')
            self.train_spectra = [str(path) for path in list(self.train_path.glob('*/*'))]
            self.train_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.train_spectra]
            self.spectra_arr = self.train_spectra
            self.label_arr = self.le.fit_transform(self.train_label)
            joblib.dump(self.le.classes_, 'class_label.pkl')
        elif mode == 'valid':
            self.valid_path = pathlib.Path(self.spectra_path + '/valid')
            self.valid_spectra = [str(path) for path in list(self.valid_path.glob('*/*'))]
            self.valid_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.valid_spectra]
            self.spectra_arr = self.valid_spectra
            self.label_arr = self.le.fit_transform(self.valid_label)
        elif mode == 'test':
            self.test_path = pathlib.Path(self.spectra_path + '/test')
            self.test_spectra = [str(path) for path in list(self.test_path.glob('*/*'))]
            self.test_label = [pathlib.Path(sing_spectra).parent.name for sing_spectra in self.test_spectra]
            self.spectra_arr = self.test_spectra
            self.label_arr = self.le.fit_transform(self.test_label)
            
        self.real_len = len(self.spectra_arr)
        
        print('Finished reading the {} set of Spectra Dataset ({} samples found)'.format(mode, self.real_len))
        
    def __getitem__(self, index):
        sing_spectra_path = self.spectra_arr[index]
        # 读取光谱
        x, y = np.loadtxt(sing_spectra_path, dtype=float, comments='#', delimiter=',', unpack=True)
        y = y.reshape(1, y.shape[0])
        y = torch.tensor(y, dtype=torch.float32)
        label = self.label_arr[index]
        
        return y, label
    
    def __len__(self):
        return self.real_len


