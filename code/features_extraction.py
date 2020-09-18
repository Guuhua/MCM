# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import numpy as np
import pandas as pd
import csv

File_Path = "C:/Users/10559/Desktop/B/特征选择/附件一_325_RON.csv"
File_write = "C:/Users/10559/Desktop/B/特征选择/Distribution.csv"

sheet_1 = pd.read_csv(File_Path, skiprows= 0)

# define dataset
X = []
for i in range(1, 366):
    if str(i) in sheet_1:
        X.append(list(sheet_1[str(i)]))
X = np.array(X)
X = X.transpose()
# print(type(X), X.dtype, X.size, X.shape, X.ndim)
y = np.array(sheet_1['RON'])

# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance

out = []
for i,v in enumerate(importance):
    out.append([list(sheet_1)[i+3], v])
    # print('Feature: %s, Score: %.5f' % (list(sheet_1)[i+3],v))

# print(out)
with open(File_write, 'w', encoding = "utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(out)

# plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.xlabel('Features')
# pyplot.ylabel('Importance')
# pyplot.title("Distribution of feature importance")
# pyplot.show()