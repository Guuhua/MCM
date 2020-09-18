import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler

File_Path_data = "C:/Users/10559/Desktop/B/model/附件一_325_RON.csv"
File_features = "C:/Users/10559/Desktop/B/model/Distribution_chosen.csv"

# get data from csv
def get_csv_data(Filepath_data='', File_features=''):
    data = pd.read_csv(Filepath_data)
    features = pd.read_csv(File_features, names=range(0,2))
    X = []
    for feature in features[0]:
        X.append(data[str(feature)].values)
    X = np.array(X)
    X = X.transpose()
    Y = data['RON'].values

    return X, Y

# generate train data
def generate_data(rate_train = 0.6, rate_val = 0.2, rate_test = 0.2):
    X, Y = get_csv_data(File_Path_data, File_features)
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
def svr_model(C = 1e0, gamma = 0.005):
    # get data
    train_x, train_y, val_x, val_y, test_x, test_y = generate_data()
    # reshape y
    train_y = np.reshape(train_y, (-1, 1))
    val_y = np.reshape(val_y, (-1, 1))
    test_y = np.reshape(test_y, (-1, 1))

    # print(train_x.shape, val_x.shape, test_x.shape)
    # print(train_y.shape, val_y, test_y)

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
    # print(test_y.shape,y_pred_test.shape)
    return rsme_val, rsme_test

print(svr_model())