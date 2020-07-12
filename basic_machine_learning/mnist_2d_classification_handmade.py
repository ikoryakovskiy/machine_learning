#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:39:08 2020

@author: ivan
"""
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

# Load and pre-process training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = (x_test / 255).reshape((-1, 28, 28, 1))
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Hyperparameters
batch_size = 128
epochs = 50
optimizer = Adam(lr=0.001)
weight_init = RandomNormal()


# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=weight_init, input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weight_init))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=weight_init))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer=weight_init))


# @tf.function
def step(real_x, real_y):
    with tf.GradientTape() as tape:
        # Make prediction
        pred_y = model(real_x.reshape((-1, 28, 28, 1)))
        # pred_y = model(tf.reshape(real_x, (-1, 28, 28, 1)))  # To use tf.function
        # Calculate loss
        model_loss = tf.keras.losses.categorical_crossentropy(real_y, pred_y)

    # Calculate gradients
    model_gradients = tape.gradient(model_loss, model.trainable_variables)
    # Update model
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))


# Training loop
bat_per_epoch = math.floor(len(x_train) / batch_size)
for epoch in range(epochs):
    print('=', end='')
    for i in range(bat_per_epoch):
        n = i * batch_size
        step(x_train[n: n + batch_size], y_train[n: n + batch_size])

# Calculate accuracy (just for evaluation)
model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
print('\n', model.evaluate(x_test, y_test, verbose=0)[1])
