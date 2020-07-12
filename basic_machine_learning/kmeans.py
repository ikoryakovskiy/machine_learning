#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:51:09 2020

@author: ivan

Based on:
    http://bjlkeng.github.io/posts/the-expectation-maximization-algorithm/
    https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(class1, class2, shape):
    ax = plt.subplot(shape)
    ax.scatter(class1[:, 0], class1[:, 1])
    ax.scatter(class2[:, 0], class2[:, 1])
    return ax


def generate_data(n):
    mean = np.array([-3, -3])
    cov = np.array([[2, 0.5], [0.5, 1]])
    class1 = np.random.multivariate_normal(mean, cov, n)

    mean = np.array([3, 0])
    cov = np.array([[1, -0.5], [-0.5, 2]])
    class2 = np.random.multivariate_normal(mean, cov, n)

    scatter_plot(class1, class2, 121)

    data = np.vstack((class1, class2))
    np.random.shuffle(data)
    return data


data = generate_data(500)

# initialize two random means
mean1 = data[0]
mean2 = data[1]

while True:
    prev_mean1 = mean1
    # calculate responsibilities
    resp1 = np.linalg.norm(data - mean1, axis=1)
    resp2 = np.linalg.norm(data - mean2, axis=1)

    class1 = data[resp1 <= resp2]
    class2 = data[resp1 > resp2]

    mean1 = np.mean(class1, axis=0)
    mean2 = np.mean(class2, axis=0)

    if np.linalg.norm(mean1 - prev_mean1) < 0.001:
        break


ax = scatter_plot(class1, class2, 122)
ax.scatter(mean1[0], mean1[1])
ax.scatter(mean2[0], mean2[1])
