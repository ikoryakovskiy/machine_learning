#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:25:54 2020

@author: ivan

https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#72a3
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped


# >> MAIN << #
def init():
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
    # normalize the features using MinMaxScalar from sklearn.preprocessing
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # could not see any improvement
#    remove_correlated_features(X)
#    remove_less_significant_features(X, Y)

    # insert intercept
    X.insert(loc=len(X.columns), column='intercept', value=1)
    # random_state is the seed used by the random number generator
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    return X_train, X_test, y_train, y_test


def calculate_loss_active(x, y, w, lmbd):
    loss = 0
    active = np.ones_like(y)
    for dp, label, i in zip(x, y, range(y.size)):
        hinge_arg = 1 - label * np.matmul(w, dp.transpose())
        if hinge_arg < 0:
            hinge_arg = 0
            active[i] = 0
        loss += hinge_arg
    loss = loss / y.size + lmbd * np.matmul(w, w.transpose())
    return loss, active


def calculate_gradient(x, y, w, lmbd, active):
    grad = lmbd * w
    grad[-1] = 0
    for dp, label, i in zip(x, y, range(y.size)):
        grad += - active[i] * label * dp
    return grad


def fit(x, y, lmbd=0.001, alpha=0.0001, tolerance=1e-4):
    w = np.random.random(size=x.shape[1]) * 2 - 1

    loss, active = calculate_loss_active(x, y, w, lmbd)
    while True:
        grad = calculate_gradient(x, y, w, lmbd, active)
        w -= alpha * grad

        new_loss, active = calculate_loss_active(x, y, w, lmbd)
        #print(new_loss)
        if abs(loss - new_loss) < tolerance:
            break
        loss = new_loss
    return w


def predict(x, w):
    hat_y = np.zeros(shape=(x.shape[0], ))

    for i, dp in enumerate(x):
        proj = np.matmul(w, dp.transpose())
        hat_y[i] = 1 if proj > 0 else -1

    return hat_y


X_train, X_test, y_train, y_test = init()

w = fit(X_train, y_train)

hat_y = predict(X_train, w)
acc = accuracy_score(y_train, hat_y)
print(f"Accuracy on the training dataset {acc}")

hat_y = predict(X_test, w)
acc = accuracy_score(y_test, hat_y)
print(f"Accuracy on the testing dataset {acc}")

cm = confusion_matrix(y_test, hat_y)
print(cm)
tn, fn, tp, fp = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]

sens = tp / (tp + fn)
spec = tn / (tn + fp)

print(f"Sens = {sens}, spec = {spec}")




