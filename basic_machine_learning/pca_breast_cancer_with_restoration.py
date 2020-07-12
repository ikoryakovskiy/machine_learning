#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:25:54 2020

@author: ivan

https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def do_pca_by_svd(data, k):
    mean = np.mean(data, axis=0)
    data = data - mean
    # Normalize because we assume:
    # - little noise (so normalization would not amplify noise)
    # - variables have different nature => scales
    # https://www.researchgate.net/post/In_which_case_data_need_to_be_normalized_before_PCA_Cluster_analysis
    std = np.std(data, axis=0)
    data = data / std
    cov = np.matmul(data.transpose(), data) / (data.shape[0] - 1)
    u, s, v = np.linalg.svd(cov)  # u and v are rotation operations: u.transpose() == v

    # sort by eigenvalue to select largest ones
    eigenval, eigenvec = s, u

    idx = np.argsort(eigenval)[::-1]
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Select the principle components
    eigenval = eigenval[:k]
    eigenvec = eigenvec[:, :k]  # take first k columns
    return np.matmul(data, eigenvec), eigenvec, mean, std


def restore_from_feature_space(original_data, compressed_data, eigenvec, mean, std):
    hat_data = np.matmul(compressed_data, eigenvec.transpose())
    hat_data = hat_data * std + mean
    compression_error = np.linalg.norm(data - hat_data) / original_data.shape[0]
    print(f"Compression error is {compression_error} per value")


data = pd.read_csv('breast-cancer-wisconsin-data/data.csv')    # SVM only accepts numerical values.
# Therefore, we will transform the categories M and B into
# values 1 and -1 (or -1 and 1), respectively.
diagnosis_map = {'M': 1, 'B': -1}
data['diagnosis'] = data['diagnosis'].map(diagnosis_map)    # drop last column (extra column added by pd)
# and unnecessary first column (id)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

# put features & outputs in different DataFrames for convenience
Y = data.loc[:, 'diagnosis']  # all rows of 'diagnosis'
X = data.iloc[:, 1:]  # all rows of column 1 and ahead (features)
data = np.array(X)

k = 2
svd_projected_data, eigenvec, mean, std = do_pca_by_svd(data, k)
class1_svd = svd_projected_data[Y == -1]
class2_svd = svd_projected_data[Y == 1]

# this one will not normalize variance
pca = PCA(k)
pca.fit(data)
pca_projected_data = pca.transform(data)
class1_pca = pca_projected_data[Y == -1]
class2_pca = pca_projected_data[Y == 1]

ax = plt.subplot("121")
ax.scatter(class1_svd[:, 0], class1_svd[:, 1])
ax.scatter(class2_svd[:, 0], class2_svd[:, 1])

ax = plt.subplot("122")
ax.scatter(class1_pca[:, 0], class1_pca[:, 1])
ax.scatter(class2_pca[:, 0], class2_pca[:, 1])

restore_from_feature_space(data, svd_projected_data, eigenvec, mean, std)
