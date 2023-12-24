import numpy as np
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from feature_selector import FeatureSelector
import pandas as pd
import matplotlib.pyplot as plt

# # # ex 1
# # input_pd = pd.read_csv('regresiya_ex1.csv', header=None)
# # input_data = input_pd.to_numpy()
# # print(input_data)

# # X = input_data[:, [1]]
# # y = input_data[:, 2]
# # print(X)
# # print(y)
# # print('Среднее xx: ', X.mean())
# # print('Среднее yy: ', y.mean())
# # model = LinearRegression()
# # model. fit (X, y)
# # print(model. intercept_ , model. coef_ , model. score (X, y))
# # # ex 1

# df = {
# "Array_1": [30, 70, 100],
# "Array_2": [65.1, 49.50, 30.7]
# }
# print("////")
# print("!!!df:")
# print(df)
# print("////")
# data = pd.DataFrame(df)

# print(data.corr())

# # ex 2
# print("ex 2")
input_pd = pd.read_csv('fish_train.csv', header=None)  
input_data = input_pd.to_numpy()
X = input_data[:, [2, 3, 4, 5, 6]]
x_0 = input_data[:, 0]
y = input_data[:, 1]
print("x_0:")
print(x_0)
print("X:")
print(X)
# df = pd.DataFrame(X)
# print("df:")
# print(df)
# print(df.corr())
# # print("y:")
# # print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=x_0)

print(f"Классы в X_train:\n{X_train}")
Widht_col = X_train[:, 4]
print('Среднее Widht_col: ', Widht_col.mean())

model = LinearRegression()
print(y_train)
model.fit(X_train, y_train)
print(model.intercept_ , model.coef_ , model.score (X_train, y_train))
y_pred = np.dot(X_test, np.array(model.coef_)) + model.intercept_
print(r2_score(y_test, y_pred))

# data = pd.read_csv('fish_train1.csv', header=None)  
# print(data.head())
# types = data.dropna(subset=['Weight'])

# correlations_data = data.corr()['Weight'].sort_values()


# # ex 2


# print("################################")
# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# # y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# print(X)
# print("/////////////////////////")
# print(y)
# reg = LinearRegression().fit(X, y)
# print(reg. intercept_ , reg. coef_ , reg. score (X, y))




