import math

from sklearn.metrics import mean_squared_error


def evaluate(train_y, train_predict, test_y, test_predict):
    '''评估模型。
    通过预测结果和真实结果比较评估模型。
    '''
    # calculate root mean squared error
    train_score = math.sqrt(mean_squared_error(train_y, train_predict))
    print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y, test_predict))
    print('Test Score: %.2f RMSE' % (test_score))
    return train_score, test_score
