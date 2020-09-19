# coding: utf-8
import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


File_Path_data = "C:/Users/10559/Desktop/B/model/附件一_325_RON.csv"
File_features = "C:/Users/10559/Desktop/B/model/Distribution_chosen.csv"


# build PSO
class PSO():
    def __init__(self, N, dim, iters, filepath):

        self.W = 0.9
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.4
        self.N  = N                               # 粒子群数量
        self.dim = dim                            # 粒子的维度
        self.iters = iters                        # 迭代优化的次数
        self.N_min = np.zeros((1, self.N))        # 粒子位置最小值
        self.N_max = np.zeros((1, self.N))        # 粒子位置最大值
        self.loc = np.zeros((self.N, self.dim))   # 粒子的位置
        self.Vel = np.zeros((self.N, self.dim))   # 粒子的速度
        self.pbest = np.zeros((self.N, self.dim)) # 个体的最佳位置
        self.gbest = np.zeros((1, self.dim))      # 全局最佳位置
        self.N_fit = np.zeros(self.N)             # 每个个体的历史最佳适应值
        self.B_fit = 1                            # 全局最佳适应值

        self.filepath_data = filepath[0]
        self.path_features = filepath[1]
        self.svr_RON = SVR(c = 3e5, gamma = 1e3, filepath = filepath, type_y = 'RON')
        self.svr_RON = SVR(c = 4e5, gamma = 9e5, filepath = filepath, type_y = 'S')
    
    # 适应函数
    def fit_function(self, X):
        return self.svr_RON.model(X)
    
    def get_features(self):
        features = pd.read_csv(self.path_features, names = range(0, 3))
        self.N_min = features[1].values
        self.N_max = features[2].values

    # 初始化粒子群
    def init(self):
        self.get_features()
        for i in range(self.N):
            for j in range(self.dim):
                self.loc[i][j] = random.uniform(self.N_min[j], self.N_max[j])
                self.Vel[i][j] = random.uniform(self.N_min[j], self.N_max[j])
            self.pbest[i] = self.loc[i]
            self.N_fit[i] = self.fit_function(self.loc[i])

            if self.N_fit[i] < self.B_fit:
                self.B_fit = self.N_fit[i]
                self.gbest = self.loc[i]

    def iterator(self):
        fitness = []

        # 迭代开始
        for t in range(self.iters):
            for i in range(self.N):
                temp = self.fit_function(self.loc[i])
                # 更新个体最优值
                if temp < self.N_fit[i]:
                    self.N_fit[i] = temp
                    self.pbest[i] = self.loc[i]

                    # 更新全局最优值
                    if self.N_fit[i] < self.B_fit:
                        self.B_fit = self.N_fit[i]
                        self.gbest = self.loc[i]

            for i in range(self.N):
                # 更新粒子的速度
                self.Vel[i] = self.W * self.Vel[i] + self.c1 * self. r1 * (self.pbest[i] - self.loc[i]) + self.c2 * self.r2 * (self.gbest - self.loc[i])
                # 更新粒子的位置
                self.loc[i] = self.loc[i] + self.Vel[i]
                for j in range(self.dim):
                    if self.loc[i][j] < self.N_min:
                        self.loc[i][j] = self.N_min
                    elif self.loc[i][j] > self.N_max:
                        self.loc[i][j] = self.N_max
            
            fitness.append(self.B_fit)
            print('iter:', t, 'RSME:', self.B_fit, _, 'C:', self.gbest[0], 'Gamma', self.gbest[1])

        return fitness

class SVR():
    def __init__(self, c, gamma, filepath, type_y):

        self.c = c
        self.gamma = gamma
        self.Filepath_data = filepath[0]
        self.File_features = filepath[1]
        self.type_y = type_y

    def get_data(self):

        data = pd.read_csv(self.Filepath_data)
        features = pd.read_csv(self.File_features, names=range(0,2))
        X = []
        for feature in features[0]:
            X.append(data[str(feature)].values)
        for i in range(1,12):
            X.append(data[str(i)].values)
        X = np.array(X)
        X = X.transpose()
        Y = data[self.type_y].values

        return X, Y
    
    def model(self, test_data):
        X, Y = self.get_data()
        
        Y = np.reshape(Y, (-1, 1))
        
        sx = StandardScaler()
        sy = StandardScaler()

        X = sx.fit_transform(X)
        Y = sy.fit_transform(Y)

        # train the svr
        regressor = svm.SVR(kernel='rbf', C = self.c, gamma = self.gamma)
        regressor.fit(X, Y.ravel())

        test_x = sx.fit_transform(test_data)
        y_pred = regressor.predict(test_x)

        y_pred = sy.inverse_transform(y_pred)

        return y_pred
        


if __name__ == '__main__':
    pass