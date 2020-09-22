import csv
import numpy as np
import pandas as pd
from pso_svr import generate_data
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# 创建多层感知机对象
def MLPR():
    train_x, train_y, _, _, test_x, test_y = generate_data(data_Y='RON')
    x_pre = preprocessing.MinMaxScaler()
    y_pre = preprocessing.MinMaxScaler()

    # 转换数据维度
    train_y = np.reshape(train_y, (-1, 1))
    test_y = np.reshape(test_y, (-1, 1))

    # features scale
    train_x = x_pre.fit_transform(train_x)
    test_x = x_pre.fit_transform(test_x)

    train_y = y_pre.fit_transform(train_y)

    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', alpha=0.1, max_iter=1000)

    model.fit(train_x, train_y.ravel())
    y_pred = model.predict(test_x)

    y_pred = np.reshape(y_pred, (-1, 1))

    y_pred = y_pre.inverse_transform(y_pred)
    test_y = test_y.flatten()
    y_pred = y_pred.flatten()

    RSME = (mean_squared_error(y_pred, test_y))**0.5
    
    return RSME, y_pred, test_y

if __name__ =='__main__':
    RSME_ave = []
    for i in range(10):
        tmp1, tmp2, tmp3 = MLPR()
        RSME_ave.append(tmp1)
    print(sum(RSME_ave)/10)
    File_write = "C:/Users/10559/Desktop/B/model/result_MLPR.csv"
    with open(File_write, 'w',newline='', encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(tmp2)
        writer.writerow(tmp3)