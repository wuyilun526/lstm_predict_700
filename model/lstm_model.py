import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def create_lstm_model(look_back):
    """构造LSTM模型。
    input-args:
        look_back: 步长值
    return:
        model: KERAS的LSTM模型
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def model_fit(model, train_x, train_y):
    """训练模型"""
    model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)


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
    plt.plot(dataset)
    plt.show()