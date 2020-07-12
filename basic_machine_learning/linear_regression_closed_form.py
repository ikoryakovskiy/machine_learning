#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:20:00 2020

@author: ivan
"""

import numpy as np


def generate_dataset(n, true_w):
    m = len(true_w)
    x = np.random.random(size=(n, m - 1))
    x_extended = np.hstack((np.ones(shape=(n, 1)), x))
    y = np.matmul(x_extended, true_w)
    y = np.reshape(y, newshape=((n, 1)))
    noisy_y = y + 0.00001 * np.random.random(size=(n, 1))
    return x, noisy_y


def linear_regression_without_regularization(x, y):
    xty = np.matmul(x.transpose(), y)
    xtx_inv = np.linalg.inv(np.matmul(x.transpose(), x))
    hat_y = np.matmul(xtx_inv, xty)
    return hat_y


def linear_regression_with_regularization(x, y, lmbd):
    n, m = x.shape
    xty = np.matmul(x.transpose(), y) / n
    xtx_lmbd_inv = np.linalg.inv(np.matmul(x.transpose(), x) / n + lmbd * np.identity(m))
    hat_y = np.matmul(xtx_lmbd_inv, xty)
    return hat_y


def main():
    # The effect of regularization is especially pronounced in case when n < m, or when there is
    # a lot of linear dependency between values
    n = 5
    x, y = generate_dataset(n, [1, 2, 3, 5, 6, 7])
    x_extended = np.hstack((np.ones(shape=(n, 1)), x))
    hat_y_no_reg = linear_regression_without_regularization(x_extended, y)
    hat_y_reg = linear_regression_with_regularization(x_extended, y, lmbd=0.001)

    print(hat_y_no_reg)
    print(hat_y_reg)


if __name__ == "__main__":
    main()
