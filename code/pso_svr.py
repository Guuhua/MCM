import csv
import random
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


File_Path_data = "C:/Users/10559/Desktop/B/model/附件一_325_RON.csv"
File_features = "C:/Users/10559/Desktop/B/model/Distribution_chosen.csv"


# 创建PSO粒子群优化模型对象
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
        self.N_min = [1e0, 1e-5]                  # 粒子位置最小值
        self.N_max = [1e1, 1e-1]                  # 粒子位置最大值
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
                self.loc[i][j] = random.uniform(self.N_min[j], self.N_max[j])
                self.Vel[i][j] = random.uniform(self.N_min[j], self.N_max[j])
            self.pbest[i] = self.loc[i]
            self.N_fit[i], _ = self.fit_function(self.loc[i])

            if self.N_fit[i] < self.B_fit:
                self.B_fit = self.N_fit[i]
                self.gbest = self.loc[i]

    # 迭代找寻最优值
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
                    if self.loc[i][j] < self.N_min[j]:
                        self.loc[i][j] = random.uniform(self.N_min[j], self.N_max[j])
                    elif self.loc[i][j] > self.N_max[j]:
                        self.loc[i][j] = random.uniform(self.N_min[j], self.N_max[j])
            
            fitness.append(self.B_fit)
            print('iter:', t, 'RSME:', self.B_fit, _, 'C:', self.gbest[0], 'Gamma', self.gbest[1])

        return fitness

# 从csv文件中读数据
def get_csv_data(Filepath_data='', File_features='', data_Y=''):
    data = pd.read_csv(Filepath_data)
    features = pd.read_csv(File_features, names=range(0,3))
    X = []
    for feature in features[0]:
        X.append(data[str(feature)].values)
    for i in range(1,12):
        X.append(data[str(i)].values)
    X = np.array(X)
    X = X.transpose()
    Y = data[data_Y].values
    return X, Y

# 生成训练数据
def generate_data(rate_train = 0.6, rate_val = 0.2, rate_test = 0.2, data_Y=''):
    X, Y = get_csv_data(File_Path_data, File_features, data_Y= data_Y)
    length = int(len(Y))
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
        elif i < int(length*(rate_test+rate_train)):
            test_x.append(X[N[i]])
            test_y.append(Y[N[i]])
        else:
            val_x.append(X[N[i]])
            val_y.append(Y[N[i]])
            
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y

def generate_data_0(rate_train = 0.6, rate_val = 0.2, rate_test = 0.2, data_Y=''):
    X, Y = get_csv_data(File_Path_data, File_features, data_Y= data_Y)
    length = int(len(Y)*(rate_train+rate_val))
    N = list(range(length))
    random.shuffle(N)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    for i in range(length, len(Y)):
        test_x.append(X[i])
        test_y.append(Y[i])
    for i in range(length):
        if i < int(length*rate_train):
            train_x.append(X[N[i]])
            train_y.append(Y[N[i]])
        else:
            val_x.append(X[N[i]])
            val_y.append(Y[N[i]])
            
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y

# 训练svr模型
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

    # File_write = "C:/Users/10559/Desktop/B/model/result_pso_svr.csv"
    # with open(File_write, 'w',newline='', encoding = "utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(y_pred_test)
    #     writer.writerow(test_y)

    val_y = val_y.flatten()
    rsme_val  = (sum((val_y-y_pred_val)**2)/len(y_pred_val)) ** 0.5
    rsme_test = (sum((test_y-y_pred_test)**2)/len(y_pred_test)) ** 0.5
    return rsme_val, rsme_test

if __name__ == '__main__':

    # ori_x = [1e2, 0.01]
    # pso_x = [1e0, 0.03]
    # print(svr_model(X=ori_x))

    svr_pso = PSO(N = 20, dim = 2, iters = 30, data_Y= 'RON')
    svr_pso.init()
    fitness = svr_pso.iterator()
    RON_X = svr_pso.gbest

    # 可视化训练过程
    plt.figure(1)
    plt.title("training process")
    plt.xlabel("iters", size=14)
    plt.ylabel("RSME", size=14)
    t = np.array([t for t in range(0, 30)])
    fitness = np.array(fitness)
    plt.plot(t, fitness, color='orangered', linewidth=3)
    plt.show()