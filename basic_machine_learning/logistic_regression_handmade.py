#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:15:55 2020

@author: ivan

On regression and regularization:
    http://bjlkeng.github.io/posts/a-probabilistic-view-of-regression/
    http://bjlkeng.github.io/posts/probabilistic-interpretation-of-regularization/
"""
import numpy as np


def calculate_gradient(x, y, theta, regularizer):
    grad = np.zeros_like(theta)
    for dp, label in zip(x, y):
        power = - np.dot(theta, dp.transpose())[0]
        grad -= dp * (label - 1 / (1 + np.exp(power)))

    reg_theta = np.hstack((np.zeros((1, 1)), theta[:, 1:]))
    return grad + regularizer * reg_theta


def calculate_loss(x, y, theta):
    loss = 0
    for dp, label in zip(x, y):
        power = - np.dot(theta, dp.transpose())[0]
        loss += - label * np.log(1 / (1 + np.exp(power))) - (1 - label) * np.log(1 - 1 / (1 + np.exp(power)))
    return loss


def train(x, y, regularizer):
    theta = np.random.normal(size=(1, x.shape[1]))
    prev_theta = theta
    alpha = 0.01
    while True:
        grad = calculate_gradient(x, y, theta, regularizer)
        prev_theta = theta
        theta = prev_theta - alpha * grad

        loss = calculate_loss(x, y, theta)
        change = np.linalg.norm(theta - prev_theta)
        print(f"Change in parameters {change} and loss {loss}")
        if change < 1e-4:
            break
    return theta


def predict(data, theta):
    labels = []
    for dp in data:
        power = - np.dot(theta, dp.transpose())[0]
        probability = 1 / (1 + np.exp(power))
        labels.append(1 if probability > 0.5 else 0)
    return labels


# First two digits are X, last one is label y
# Rule: sum(X) > 7 => class 1, else class 0
train_data = [
    [2, 3, 0],
    [3, 2, 0],
    [1, 4, 0],
    [3, 8, 1],
    [8, 0, 1],
    [13, 2, 1],
    [10, 3, 1],
    [0, 5, 0],
    [7, 0, 0],
    [6, 2, 1],
]
train_data = np.array(train_data)
train_x, train_y = train_data[:, :-1], train_data[:, -1]
train_x = np.hstack((np.ones(shape=(train_x.shape[0], 1)), train_x))
theta = train(train_x, train_y, regularizer=0.1)

print(f"Obtained theta is {theta}")

test_x = [
    [0, 0],  # 0
    [3, 4],  # 0
    [10, 0],  # 1
    [5, 5],  # 1
    [2, 3],  # 0
]
test_x = np.array(test_x)
test_x = np.hstack((np.ones(shape=(test_x.shape[0], 1)), test_x))

pred_train_y = predict(train_x, theta)
print(f"Predicted labels on the training dataset {pred_train_y}")

pred_test_y = predict(test_x, theta)
print(f"Predicted labels on the testing dataset {pred_test_y}")
