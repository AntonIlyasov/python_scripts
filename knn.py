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
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# ex 1
# input_pd = pd.read_csv('knn_ex1.csv', header=None)
# input_data = input_pd.to_numpy()
# print(input_data)

# y = input_data[:, 3]
# X = input_data[:, [1, 2]]

# neigh = KNeighborsClassifier(n_neighbors=3, metric='cityblock')
# neigh.fit(X, y)
# print(neigh.predict([[74, 92]]))
# print(neigh.kneighbors([[74, 92]]))

# ex 2
raw = pd.read_csv("adult_data_train.csv")
pred=raw.drop(['education','marital-status'],axis=1)
print(pred)
for c in raw.columns:
    raw[c] = raw[c].apply(lambda x: raw[c].mode()[0] if x == '?' else x)

