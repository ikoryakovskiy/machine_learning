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


def do_pca_by_eig(data, k):
    mean = np.mean(data, axis=0)
    data = data - mean
    std = np.std(data, axis=0)
    data = data / std
    cov = np.matmul(data.transpose(), data) / (data.shape[0] - 1)

    # The eigenvectors represent the directions or components for the reduced subspace of A,
    # whereas the eigenvalues represent the magnitudes for the directions.
    # Note: covariance matrix is symmetric.Therefore, column ``v[:,i]`` is the eigenvector
    # corresponding to the eigenvalue ``w[i]`` (which holds the transposed cov, obviously)
    eigenval, eigenvec = np.linalg.eig(cov)

    # sort by eigenvalue to select largest ones
    idx = np.argsort(eigenval)[::-1]
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Select the principle components
    eigenval = eigenval[:k]
    eigenvec = eigenvec[:, :k]  # take first k columns

    return np.matmul(data, eigenvec)


def do_pca_by_svd(data, k):
    mean = np.mean(data, axis=0)
    data = data - mean
    std = np.std(data, axis=0)
    data = data / std
    cov = np.matmul(data.transpose(), data) / (data.shape[0] - 1)
    u, s, v = np.linalg.svd(cov)

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

# Show in 2D that is seems like a classifier can help
eig_projected_data = do_pca_by_eig(data, k=2)
class1 = eig_projected_data[Y == -1]
class2 = eig_projected_data[Y == 1]

svd_projected_data, eigenvec, mean, std = do_pca_by_svd(data, k=2)
# check compression error
restore_from_feature_space(data, svd_projected_data, eigenvec, mean, std)

class1_svd = svd_projected_data[Y == -1]
class2_svd = svd_projected_data[Y == 1]

ax = plt.subplot("121")
ax.scatter(class1[:, 0], class1[:, 1])
ax.scatter(class2[:, 0], class2[:, 1])


ax = plt.subplot("122")
ax.scatter(class1_svd[:, 0], class1_svd[:, 1])
ax.scatter(class2_svd[:, 0], class2_svd[:, 1])
