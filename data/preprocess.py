import numpy as np
import keras
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import db


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


def create_lstm_model(look_back):
    """构造LSTM模型。
    input-args:
        look_back: 步长值
    return:
        model: KERAS的LSTM模型
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def model_fit(model, train_x, train_y):
    """训练模型"""
    model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)



def evaluate(model, train_x, train_y, test_x, test_y, scaler):
    '''评估模型。
    输入训练数据和测试数据，模型输出预测结果。通过预测结果和真实结果比较评估模型。

    args:
        model: lstm模型
        train_x: 训练数据的特征
        train_y: 训练数据的标签
        test_x: 测试数据的特征
        test_y: 测试数据的标签
        scaler: 做缩放的对象，用于将归一化的数据还原
    returns:
        二元组，(train_predict, test_predict)
        train_predict: 训练数据的预测结果
        test_predict: 测试数据的预测结果
    '''
    # make predictions
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % (train_score))
    testScore = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % (test_score))
    return train_predict, test_predict




def draw(dataset, train_predict, test_predict, look_back):
    '''绘制曲线查看预测效果。

    args:
        dataset: 经过缩放后的数据集
        train_predict: 训练数据的预测结果
        test_predict: 测试数据的预测结果
        look_back: 步长
    returns:
        绘制原始曲线和预测曲线。
    '''
    # shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict)+(look_back*2): len(dataset), :] = test_predict
    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.show()


if __name__ == '__main__':

    # create_record_table()
    begin_day = '2004-06-16'
    end_day = '2004-12-29'
    df = db.read_data(begin_day, end_day)

    # print(df['open_price'])
    data = df['open_price']

    # datas = np.array(df['open_price'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = np.array(data)
    dataset = np.reshape(dataset, (dataset.shape[0], 1))
    dataset = dataset.astype('float32')
    dataset = scaler.fit_transform(dataset)

    train, test = split_data(dataset)
    look_back = 5
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    # print(test_x)
    # print(test_x.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    # print(test_x.shape)
    model = create_lstm_model(look_back)
    model_fit(model, train_x, train_y)
    # evaluate(model, train_x, train_y, test_x, test_y, scaler)

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    dataset = scaler.inverse_transform(dataset)
    draw(dataset, train_predict, test_predict, look_back)

