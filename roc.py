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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def softmax(x, axis=0):
    return np.exp(x)/np.sum(np.exp(x), axis=axis).reshape((-1,1))


def my_roc_auc(y_true, y_score):
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

    tpr = tp / tp[-1]
    fpr = fp / fp[-1]

    # piece-wise integration
    area = 0
    for i in range(tpr.shape[0]-1):
        area += (fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])/2

    return fpr, tpr, area


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

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fprc = dict()
tprc = dict()
roc_aucc = dict()

fprm = dict()
tprm = dict()
roc_aucm = dict()

fpri = dict()
tpri = dict()
roc_auci = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    # softmax function applied to scores
    fprc[i], tprc[i], _ = roc_curve(y_test[:, i], y_score_softmax[:, i])
    roc_aucc[i] = auc(fprc[i], tprc[i])

    # my calculation of ROC curve
    fprm[i], tprm[i], roc_aucm[i] = my_roc_auc(y_test[:, i], y_score_softmax[:, i])

    # ideal calculation
    fpri[i], tpri[i], roc_auci[i] = my_roc_auc(y_test[:, i], y_test[:, i])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

plt.plot(fprc[2], tprc[2], color='darkgreen',
         lw=lw, label='Softmax ROC curve (area = %0.2f)' % roc_aucc[2])

plt.plot(fprm[2], tprm[2], color='yellow', linestyle='--',
         lw=lw, label='My ROC curve (area = %0.2f)' % roc_aucm[2])

plt.plot(fpri[2], tpri[2], color='black',
         lw=lw, label='Ideal ROC curve (area = %0.2f)' % roc_auci[2])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()