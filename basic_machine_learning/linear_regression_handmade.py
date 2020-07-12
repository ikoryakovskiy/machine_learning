#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:46:37 2020

@author: ivan
https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22

Why to use GradientTape?
https://stackoverflow.com/questions/53953099/what-is-the-purpose-of-the-tensorflow-gradient-tape?noredirect=1&lq=1
With eager execution enabled, Tensorflow will calculate the values of tensors as they occur in your code. This means
that it won't precompute a static graph for which inputs are fed in through placeholders. This means to back propagate
errors, you have to keep track of the gradients of your computation and then apply these gradients to an optimiser.

This is very different from running without eager execution, where you would build a graph and then simply use sess.run
to evaluate your loss and then pass this into an optimiser directly.

Fundamentally, because tensors are evaluated immediately, you don't have a graph to calculate gradients and so you need
a gradient tape. It is not so much that it is just used for visualisation, but more that you cannot implement a
gradient descent in eager mode without it.

Obviously, Tensorflow could just keep track of every gradient for every computation on every tf.Variable. However, that
could be a huge performance bottleneck. They expose a gradient tape so that you can control what areas of your code
need the gradient information. Note that in non-eager mode, this will be statically determined based on the
computational branches that are descendants of your loss but in eager mode there is no static graph and so no way of
knowing.
"""

import random
import numpy as np
import tensorflow as tf


# Loss function
def loss(real_y, pred_y):
    return tf.abs(real_y - pred_y)


# Training data
x_train = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.asarray([i * 10 + 5 for i in x_train])  # y = 10x+5

# Trainable variables
a = tf.Variable(random.random(), trainable=True)
b = tf.Variable(random.random(), trainable=True)


# Step function
def step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Make prediction
        pred_y = a * real_x + b
        # Calculate loss
        reg_loss = loss(real_y, pred_y)

    # Calculate gradients
    a_gradients, b_gradients = tape.gradient(reg_loss, (a, b))

    # Update variables
    a.assign_sub(a_gradients * 0.001)
    b.assign_sub(b_gradients * 0.001)


# Training loop
for _ in range(10000):
    step(x_train, y_train)

print(f'y ≈ {a.numpy()}x + {b.numpy()}')
