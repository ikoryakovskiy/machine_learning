#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:55:34 2020

@author: ivan
"""
from sklearn import tree
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools


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


(x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1)

x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))
x_val = np.reshape(x_val, newshape=(x_val.shape[0], -1))
x_test = np.reshape(x_test, newshape=(x_test.shape[0], -1))

dtc = tree.DecisionTreeClassifier()

dtc = dtc.fit(x_train, y_train)

y_hat = dtc.predict(x_test)

cm = confusion_matrix(y_test, y_hat)
print(classification_report(y_test, y_hat))

plot_confusion_matrix(cm, classes=range(10))

plt.show()
