import numpy as np


def split_data(dataset, split_rate=0.6):
    """数据预处理"""
    train_size = int(len(dataset) * split_rate)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:, :]
    return train, test


def create_dataset(dataset, look_back=1):
    """构造LSTM训练数据。
    input-args:
        dataset: 时序数据数组
        look_back: 步长值
    return:
        元组，(x_array, y_array)
    """
    x, y = [], []
    for i in range(len(dataset) - look_back):
        t_x = dataset[i:(i+look_back)]
        t_y = dataset[i+look_back]
        x.append(t_x)
        y.append(t_y)
    return np.array(x), np.array(y)


