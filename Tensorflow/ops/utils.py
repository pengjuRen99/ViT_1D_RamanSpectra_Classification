import pathlib
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


def get_spectra_and_labels(dataset_dir):
        data_root = pathlib.Path(dataset_dir)
        all_spectra_path = [str(path) for path in list(data_root.glob('*/*'))]
        
        '''单编码
        # print(all_spectra_path)
        # get labels' names
        label_names = sorted(item.name for item in data_root.glob('*/'))

        # dic: {label: index}
        label_to_index = dict((label, index) for index, label in enumerate(label_names))
        # get all spectras' labels
        all_spectra_label = [label_to_index[pathlib.Path(single_spectra_path).parent.name] for single_spectra_path in all_spectra_path]
        # print(all_spectra_label)
        return all_spectra_path, all_spectra_label
        '''

        # one-hot 编码
        label_names = []
        data = []
        for single_spectra_path in all_spectra_path:
            name = pathlib.Path(single_spectra_path).parent.name
            label_names.append(name)
            x, y = np.loadtxt(single_spectra_path, dtype=float, comments='#', delimiter=',', unpack=True)
            data.append(y)
        data = np.array(data)
        data = tf.reshape(data, [data.shape[0], data.shape[1], 1])
        lb = LabelBinarizer()
        all_spectra_label = lb.fit_transform(label_names)
        return data, all_spectra_label, lb


def generate_ds(dataset_dir: str, batch_size: int = 16, cache_data: bool = False):
    # 生成训练、验证和测试集
    train_data, train_label, lb = get_spectra_and_labels(dataset_dir + '/train')
    val_data, val_label, lb = get_spectra_and_labels(dataset_dir + '/valid')
    test_data, test_label, lb = get_spectra_and_labels(dataset_dir + '/test')

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_data), tf.constant(train_label)))
    total_train = len(train_data)
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_data), tf.constant(val_label)))
    total_val = len(val_data)
    val_ds = configure_for_performance(val_ds, total_val, shuffle=True, cache=cache_data)

    test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(test_data), tf.constant(test_label)))
    total_test = len(test_data)
    test_ds = configure_for_performance(test_ds, total_test, shuffle=True, cache=cache_data)

    return train_ds, val_ds, test_ds, lb


    


        

