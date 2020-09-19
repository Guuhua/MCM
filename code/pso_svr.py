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
    def __init__(self, N, dim, iters, data_Y):

        self.W = 0.9
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.4
        self.data_y = data_Y
        self.N  = N                               # 粒子群数量
        self.dim = dim                            # 粒子的维度
        self.iters = iters                        # 迭代优化的次数
        self.N_min = 1e-2                         # 粒子位置最小值
        self.N_max = 1e5                          # 粒子位置最大值
        self.loc = np.zeros((self.N, self.dim))   # 粒子的位置
        self.Vel = np.zeros((self.N, self.dim))   # 粒子的速度
        self.pbest = np.zeros((self.N, self.dim)) # 个体的最佳位置
        self.gbest = np.zeros((1, self.dim))      # 全局最佳位置
        self.N_fit = np.zeros(self.N)             # 每个个体的历史最佳适应值
        self.B_fit = 1                            # 全局最佳适应值
    
    # 适应函数
    def fit_function(self, X):
        return svr_model(X = X, data_Y = self.data_y)
    
    # 初始化粒子群
    def init(self):
        for i in range(self.N):
            for j in range(self.dim):
                self.loc[i][j] = random.uniform(self.N_min, self.N_max)
                self.Vel[i][j] = random.uniform(self.N_min, self.N_max)
            self.pbest[i] = self.loc[i]
            self.N_fit[i], _ = self.fit_function(self.loc[i])

            if self.N_fit[i] < self.B_fit:
                self.B_fit = self.N_fit[i]
                self.gbest = self.loc[i]

    def iterator(self):
        fitness = []

        # 迭代开始
        for t in range(self.iters):
            for i in range(self.N):
                temp, _ = self.fit_function(self.loc[i])
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

# get data from csv
def get_csv_data(Filepath_data='', File_features='', data_Y=''):
    data = pd.read_csv(Filepath_data)
    features = pd.read_csv(File_features, names=range(0,2))
    X = []
    for feature in features[0]:
        X.append(data[str(feature)].values)
    for i in range(1,12):
        X.append(data[str(i)].values)
    X = np.array(X)
    X = X.transpose()
    Y = data[data_Y].values

    return X, Y

# generate train data
def generate_data(rate_train = 0.6, rate_val = 0.2, rate_test = 0.2, data_Y=''):
    X, Y = get_csv_data(File_Path_data, File_features, data_Y= data_Y)
    length = len(Y)
    N = list(range(length))
    random.shuffle(N)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    for i in range(length):
        if i < int(length*rate_train):
            train_x.append(X[N[i]])
            train_y.append(Y[N[i]])
        elif i < int(length*(rate_train+rate_val)):
            val_x.append(X[N[i]])
            val_y.append(Y[N[i]])
        else:
            test_x.append(X[N[i]])
            test_y.append(Y[N[i]])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_x = np.array(test_x)

    return train_x, train_y, val_x, val_y, test_x, test_y

# train the svr model
def svr_model(X, data_Y = 'RON'):
    C = X[0]
    gamma = X[1]
    # get data
    train_x, train_y, val_x, val_y, test_x, test_y = generate_data(data_Y= data_Y)
    # reshape y
    train_y = np.reshape(train_y, (-1, 1))
    val_y = np.reshape(val_y, (-1, 1))
    test_y = np.reshape(test_y, (-1, 1))

    # Feature Scaling
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    train_x = sc_x.fit_transform(train_x)
    val_x = sc_x.fit_transform(val_x)
    test_x = sc_x.fit_transform(test_x)

    train_y = sc_y.fit_transform(train_y)
    val_y = sc_y.fit_transform(val_y)
    test_y = sc_y.fit_transform(test_y)

    # train the svr
    regressor = svm.SVR(kernel='rbf', C = C, gamma = gamma)
    regressor.fit(train_x, train_y.ravel())

    # predicting a new result
    y_pred_test = regressor.predict(test_x)
    y_pred_val = regressor.predict(val_x)

    y_pred_test = sc_y.inverse_transform(y_pred_test)
    test_y = sc_y.inverse_transform(test_y)
    y_pred_val = sc_y.inverse_transform(y_pred_val)
    val_y = sc_y.inverse_transform(val_y)

    test_y = test_y.flatten()
    val_y = val_y.flatten()
    rsme_val  = (sum((val_y-y_pred_val)**2)/len(y_pred_val)) ** 0.5
    rsme_test = (sum((test_y-y_pred_test)**2)/len(y_pred_test)) ** 0.5
    return rsme_val, rsme_test


if __name__ == '__main__':

    # svr_model()

    svr_pso = PSO(N = 40, dim = 2, iters = 10, data_Y= 'RON')
    svr_pso.init()
    fitness = svr_pso.iterator()
    RON_X = svr_pso.gbest

    # RON_X = [3e5, 1e3]
    # S_X = [4e5, 9e5]

    plt.figure(1)
    plt.title("training process")
    plt.xlabel("iters", size=14)
    plt.ylabel("RSME", size=14)
    t = np.array([t for t in range(0, 10)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='orangered', linewidth=3)
    plt.show()