import numpy as np
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

input_pd = pd.read_csv('94_16.csv', header=None)
x = input_pd.to_numpy()
print(x)

print('//////////////////////')
pca = PCA(n_components=2, svd_solver='full')
pca_features = pca.fit_transform(x)

print('Shape before PCA: ', x.shape)
print('Shape after PCA: ', pca_features.shape)

pca_df = pd.DataFrame(
  data=pca_features, 
  columns=[
    'Principal Component 1', 
    'Principal Component 2'
    ])

print(pca_df)
print(pca.explained_variance_ratio_)


# pca_df = pd.DataFrame(
#   data=pca_features, 
#   columns=[
#     'Principal Component 1', 
#     'Principal Component 2',
#     'Principal Component 3',
#     'Principal Component 4',
#     'Principal Component 5',
#     'Principal Component 6',
#     'Principal Component 7',
#     'Principal Component 8',
#     'Principal Component 9',
#     'Principal Component 10'
#     ])