#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:46:17 2020

@author: ivan
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools


def plot_examples(images, labels):
    f, axarr = plt.subplots(1, len(images))
    for ax, img, label in zip(axarr, images, labels):
        pixels = img.reshape((28, 28))
        ax.imshow(pixels, cmap='gray')
        ax.set_title(label)


def plot_traning_curves(history):
    categorical_accuracy, = plt.plot(history.history["sparse_categorical_accuracy"], label='categorical_accuracy')
    val_categorical_accuracy, = plt.plot(
        history.history["val_sparse_categorical_accuracy"], label='val_categorical_accuracy'
    )
    loss_plt, = plt.plot(history.history["loss"], label='loss')
    val_loss_plt, = plt.plot(history.history["val_loss"], label='val_loss')
    plt.legend(handles=[categorical_accuracy, val_categorical_accuracy, loss_plt, val_loss_plt])


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


model = keras.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# logits (numeric output of the last linear layer of a multi-class classification neural network)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=keras.optimizers.RMSprop(), loss=loss, metrics=['sparse_categorical_accuracy'])

filepath = "mnist-1d.hdf5"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, min_delta=0.000, verbose=1),
    ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
]

# Train
history = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.1, callbacks=callbacks, verbose=1)

# plot training curves
plot_traning_curves(history)

# Test
model.load_weights(filepath)

output_results = model.predict(x_test)
y_predict = [np.argmax(output) for output in output_results]

display_num = 4
plot_examples(x_test[:display_num], y_predict[:display_num])

cm = confusion_matrix(y_predict, y_test, labels=range(10))
plot_confusion_matrix(cm, classes=range(10))

accuracy = accuracy_score(y_predict, y_test)
print(f"Testing accuracy {accuracy}")
plt.show()
