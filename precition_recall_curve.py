#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:48:40 2019

@author: Koryakovskiy Ivan (i.koryakovskiy@gmail.com)
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def softmax(x, axis=0):
    return np.exp(x)/np.sum(np.exp(x), axis=axis).reshape((-1,1))


def my_prc(y_true, y_score):
    # Intuition:
    # 1. What kind of error do we have, when we correctly classified the first
    #    point which has the highest score (most probably correctly classified)?
    # 2. What kind of error do we have, when we correctly classified the first
    #    two points?
    # 3. What kind of error do we have, when we correctly classified the first
    #    three points?
    # 4. continue ...
    idx = np.argsort(y_score, kind='mergesort')[::-1]
    y_true = y_true[idx]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1-y_true)

    recall = tp / tp[-1]
    precision = tp / (tp+fp)

    # piece-wise integration
    area = 0
    for i in range(precision.shape[0]-1):
        area += (precision[i+1]-precision[i])*(recall[i+1]+recall[i])/2

    return precision, recall, area


# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Transformations to the score do not change ROC
#y_score_softmax = np.log(y_score - np.min(y_score)+0.0001)
#y_score_softmax = 100*(y_score - np.min(y_score))
y_score_softmax = softmax(y_score, axis=1)
#y_score_softmax = y_score

# Compute ROC curve and ROC area for each class
precisiono = dict()
recallo = dict()

precision = dict()
recall = dict()

precisionm = dict()
recallm = dict()

for i in range(n_classes):
    precisiono[i], recallo[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])

    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score_softmax[:, i])

    precisionm[i], recallm[i], _ = my_prc(y_test[:, i], y_score_softmax[:, i])


plt.figure()
lw = 2
plt.plot(recallo[2], precisiono[2], color='darkorange',
         lw=lw, label='PR Curve original')

plt.plot(recall[2], precision[2], color='darkgreen',
         lw=lw, label='PR Curve softmax')

plt.plot(recallm[2], precisionm[2], color='yellow', linestyle='--',
         lw=lw, label='PR Curve my')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precition')
plt.title('PR curve example')
plt.legend(loc="lower right")
plt.show()