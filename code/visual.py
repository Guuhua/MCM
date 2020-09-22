import csv
import random
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


File_Path_data = "C:/Users/10559/Desktop/B/visual/附件一_325_RON.csv"
File_features = "C:/Users/10559/Desktop/B/visual/Distribution_chosen.csv"

filepath = [File_Path_data, File_features]

# 创建可视化对象
class Visual():
    def __init__(self, c, gamma, filepath, type_y):

        self.c = c
        self.gamma = gamma
        self.Filepath_data = filepath[0]
        self.File_features = filepath[1]
        self.type_y = type_y

    # 获取数据
    def get_data(self):
        p = []
        X = []
        data = pd.read_csv(self.Filepath_data)
        features = pd.read_csv(self.File_features, names=range(0,4))
        
        for i in range(len(features[0])):
            p.append([features[1][i], features[2][i], features[3][i]])

        for feature in features[0]:
            X.append(data[str(feature)].values)
        for i in range(1,12):
            X.append(data[str(i)].values)
        X = np.array(X)
        X = X.transpose()
        Y = data[self.type_y].values

        return X, Y, p
    
    # 生成数据
    def gengerate_data(self, X, p, N):
        p_tmp = np.arange(p[N][0], p[N][1], p[N][2])
        test_data = []
        for i in p_tmp:
            temp = np.concatenate([X[0:N], [i], X[N+1:]])
            test_data.append(temp)
        test_data = np.array(test_data)

        return test_data, p_tmp

    # 训练模型
    def model(self):
        X, Y, p = self.get_data()

        Y = np.reshape(Y, (-1, 1))
        
        sx = StandardScaler()
        sy = StandardScaler()

        X = sx.fit_transform(X)
        Y = sy.fit_transform(Y)

        # train the svr
        regressor = svm.SVR(kernel='rbf', C = self.c, gamma = self.gamma)
        regressor.fit(X, Y.ravel())

        X = sx.inverse_transform(X)
        y_pred = []
        x_pred = []
        for N in range(15):
            test_data, xxx = self.gengerate_data(X[132], p, N)
            test_data = sx.fit_transform(test_data)
            y_p = regressor.predict(test_data)
            y_p = sy.inverse_transform(y_p)
            if self.type_y == 'RON': 
                y_p = 89.4 - y_p
            y_pred.append(y_p)
            x_pred.append(xxx)
        y_pred = np.array(y_pred)
        x_pred = np.array(x_pred)
        File_write = "C:/Users/10559/Desktop/B/visual/result_"
        with open(File_write+self.type_y+'.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(y_pred)
            writer.writerows(x_pred)
        return y_pred, x_pred

if __name__ == '__main__':
    Vis = Visual(c=1e0, gamma=0.03, filepath=filepath, type_y='RON')
    Vis_ = Visual(c=1e0, gamma=0.03, filepath=filepath, type_y='S')

    RON_y, RON_x = Vis.model()
    S_y,   S_x   = Vis_.model()
    label_X = ['id_21', 'id_27', 'id_31', 'id_33', 'id_35', 'id_53', 'id_75', \
        'id_84', 'id_131', 'id_134', 'id_177', 'id_244', 'id_257', 'id_276', 'id_365']
    # 可视化15维数据
    for i in range(len(RON_x)):
        fig, ax1 = plt.subplots()
        ax1.plot(RON_x[i], RON_y[i], color='r', linestyle='-', linewidth=1)
        ax1.set_xlabel(label_X[i])
        ax1.set_ylabel('RON')
        ax2 = ax1.twinx()
        ax2.plot(RON_x[i], S_y[i], color='b', linestyle='-', linewidth=1)
        ax2.set_ylabel('S')

        ax1.legend((u'RON'), loc='upper left')
        ax2.legend((u'S'), loc='upper right')
        plt.show()

