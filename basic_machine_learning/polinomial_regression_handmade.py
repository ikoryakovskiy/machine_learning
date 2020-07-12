#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:55:52 2020

@author: ivan
https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22
"""
import random
import numpy as np
import tensorflow as tf


# Loss function
def loss(real_y, pred_y):
    return tf.abs(real_y - pred_y)


# Training data
x_train = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.asarray([6 * i**2 + 8 * i + 2 for i in x_train])  # y = 6x^2 + 8x + 2

# Trainable variables
a = tf.Variable(random.random(), trainable=True)
b = tf.Variable(random.random(), trainable=True)
c = tf.Variable(random.random(), trainable=True)


# Step function
def step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Make prediction
        pred_y = a * real_x**2 + b * real_x + c
        # Calculate loss
        poly_loss = loss(real_y, pred_y)

    # Calculate gradients
    a_gradients, b_gradients, c_gradients = tape.gradient(poly_loss, (a, b, c))

    # Update variables
    a.assign_sub(a_gradients * 0.001)
    b.assign_sub(b_gradients * 0.001)
    c.assign_sub(c_gradients * 0.001)


# Training loop
for _ in range(10000):
    step(x_train, y_train)

print(f'y ≈ {a.numpy()}x^2 + {b.numpy()}x + {c.numpy()}')
