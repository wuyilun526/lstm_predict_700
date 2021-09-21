import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data import db, preprocess
from model import lstm_model

def run():
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

    train, test = preprocess.split_data(dataset)
    look_back = 5
    train_x, train_y = preprocess.create_dataset(train, look_back)
    test_x, test_y = preprocess.create_dataset(test, look_back)
    # print(test_x)
    # print(test_x.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    # print(test_x.shape)
    model = lstm_model.create_lstm_model(look_back)
    lstm_model.model_fit(model, train_x, train_y)
    # evaluate(model, train_x, train_y, test_x, test_y, scaler)

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    dataset = scaler.inverse_transform(dataset)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    lstm_model.draw(dataset, train_predict, test_predict, look_back)


if __name__ == '__main__':
    run()
