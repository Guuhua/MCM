import csv
import random
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


File_Path_data = "C:/Users/10559/Desktop/B/model/附件一_325_RON.csv"
File_features = "C:/Users/10559/Desktop/B/model/Distribution_chosen.csv"

filepath = [File_Path_data, File_features]

# 创建PSO优化对象
class PSO():
    def __init__(self, N, dim, iters, filepath, NT):

        self.W = 0.8
        self.c1 = 1
        self.c2 = 4
        self.r1 = 0.4
        self.r2 = 0.2
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
        self.B_fit = 5                            # 全局最佳适应值

        self.NT = NT
        self.S = 0

        self.filepath_data = filepath[0]
        self.path_features = filepath[1]
        self.svr_RON = SVR(c = 1e1, gamma = 1e-3, filepath = filepath, type_y = 'RON')
        self.svr_S = SVR(c = 1e1, gamma = 1e-3, filepath = filepath, type_y = 'S')
    
    def get_data(self, X):
        data = pd.read_csv(self.filepath_data)
        d11 = []
        for j in X:
            d11.append(j)

        for i in range(1, 12):
            d11.append(data[str(i)][self.NT])
        
        return np.array(d11)

    # 适应函数
    def fit_function(self, X):
        X = self.get_data(X)
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
                tmp = self.svr_S.model(self.get_data(self.loc[i]))
                if tmp <= 5:
                    self.S = tmp
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
                        tmp = self.svr_S.model(self.get_data(self.loc[i]))
                        if tmp <= 5:
                            self.S = tmp
                            self.B_fit = self.N_fit[i]
                            self.gbest = self.loc[i]

            for i in range(self.N):
                # 更新粒子的速度
                self.Vel[i] = self.W * self.Vel[i] + self.c1 * self. r1 * (self.pbest[i] - self.loc[i]) + self.c2 * self.r2 * (self.gbest - self.loc[i])
                # 更新粒子的位置
                self.loc[i] = self.loc[i] + self.Vel[i]
                for j in range(self.dim):
                    if self.loc[i][j] < self.N_min[j]:
                        self.loc[i][j] = self.N_min[j]
                    elif self.loc[i][j] > self.N_max[j]:
                        self.loc[i][j] = self.N_max[j]
            
            fitness.append(self.B_fit)
            print('iter:', t, 'RON:', self.B_fit, 'S:', self.S)

        return fitness

# 创建SVR对象
class SVR():
    def __init__(self, c, gamma, filepath, type_y):

        self.c = c
        self.gamma = gamma
        self.Filepath_data = filepath[0]
        self.File_features = filepath[1]
        self.type_y = type_y

    # 获取数据
    def get_data(self):
        data = pd.read_csv(self.Filepath_data)
        features = pd.read_csv(self.File_features, names=range(0,3))
        X = []
        for feature in features[0]:
            X.append(data[str(feature)].values)
        for i in range(1,12):
            X.append(data[str(i)].values)
        X = np.array(X)
        X = X.transpose()
        Y = data[self.type_y].values

        return X, Y
    
    # 训练模型
    def model(self, test_data):
        X, Y = self.get_data()

        test_data = np.reshape(test_data, (1,-1))
        X1 = np.vstack((X, test_data))
        Y = np.reshape(Y, (-1, 1))
        
        sx = StandardScaler()
        sy = StandardScaler()

        X1 = sx.fit_transform(X1)

        X = X1[0:-1]
        test_data = X1[-1]
        test_data = np.reshape(test_data, (1,-1))

        Y = sy.fit_transform(Y)

        # train the svr
        regressor = svm.SVR(kernel='rbf', C = self.c, gamma = self.gamma)
        regressor.fit(X, Y.ravel())

        y_pred = regressor.predict(test_data)
        y_pred = sy.inverse_transform(y_pred)

        return y_pred[0]


if __name__ == '__main__':
    best_loc = []
    best_s = []
    best_RON = []
    for i in range(50):
        RON_PSO = PSO(10, 15, 30, filepath, i)
        RON_PSO.init()
        RON_PSO.iterator()
        best_loc.append(RON_PSO.gbest)
        best_s.append(RON_PSO.S)
        best_RON.append(RON_PSO.B_fit)
    File_write = "C:/Users/10559/Desktop/B/model/15dim_p.csv"
    with open(File_write, 'w',newline='', encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(best_loc)
    File_write = "C:/Users/10559/Desktop/B/model/15dim_s.csv"
    with open(File_write, 'w',newline='', encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(best_s)
    File_write = "C:/Users/10559/Desktop/B/model/15dim_RON.csv"
    with open(File_write, 'w',newline='', encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(best_RON)    

    