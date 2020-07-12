#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:59:37 2020

@author: ivan
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def compute_feature(x0, x1):
    return np.array([1, x0, x1, x0 * x1, x0**2, x1**2])


def generate_dataset(sigma=0.0):
    samples_per_dim = 21
    dataset_x = []
    dataset_y = []
    beta = np.array([[0.7, 0.1, -3, -2.5, 1, 2.7], [-0.3, -0.1, 0, 2.1, -1.1, 0.7], [1.7, 2.1, -3.5, -0.5, 1.7, -2.0]])
    for x0 in np.linspace(-1, 1, samples_per_dim):
        for x1 in np.linspace(-1, 1, samples_per_dim):
            feature_x = compute_feature(x0, x1)
            y = np.dot(beta, feature_x) + np.random.normal(loc=0, scale=sigma, size=(3,))
            dataset_x.append((x0, x1))
            dataset_y.append(y)
    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    return dataset_x, dataset_y


def get_features(dataset_x):
    features = []
    for x in dataset_x:
        features.append(compute_feature(x[0], x[1]))
    features = np.array(features)
    return features


def create_model(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(Dense(output_size, input_shape=(input_size,)))
    return model


if __name__ == "__main__":
    tf.keras.backend.clear_session()

    dataset_x, dataset_y = generate_dataset()

    features = get_features(dataset_x)
    x_train_val, x_test, y_train_val, y_test = train_test_split(features, dataset_y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1)

    model = create_model(input_size=6, output_size=3)
    loss = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    model.compile(optimizer=optimizer, loss=loss)

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, min_delta=0.000, verbose=1)]

    # For debugging, print trainable variables
    tv = model.layers[0].trainable_variables
    print(tv)

    history = model.fit(
        x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=callbacks
    )

    loss_plt, = plt.plot(history.history["loss"], label='loss')
    val_loss_plt, = plt.plot(history.history["val_loss"], label='val_loss')
    plt.legend(handles=[loss_plt, val_loss_plt])

    print('# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    weights = model.layers[0].get_weights()
    print(weights)
    plt.show()
