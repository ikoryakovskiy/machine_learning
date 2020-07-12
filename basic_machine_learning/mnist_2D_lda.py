# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:24:41 2016

@author: SeanEaster
"""

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
import numpy as np


def untick(sub):
    sub.tick_params(
        which='both', bottom='off', top='off', labelbottom='off', labelleft='off', left='off', right='off'
    )

# Each image is 8x8 pixels.
# Text -- each image,
# Words -- pixels of a grayscale intensity
# => if we have 10 topics, then each topic will corespond to each image of a number.


digits = load_digits()
images = digits['images']
images = [image.reshape((1, -1)) for image in images]
images = np.concatenate(tuple(images), axis=0)

topicsRange = [i + 4 for i in range(22)]

ldaModels = [LDA(n_components=numTopics) for numTopics in topicsRange]

for lda in ldaModels:
    lda.fit(images)

scores = [lda.score(images) for lda in ldaModels]

plt.plot(topicsRange, scores)
plt.show()

maxLogLikelihoodTopicsNumber = np.argmax(scores)
plotNumbers = [4, 9, 16, 25]

if maxLogLikelihoodTopicsNumber not in plotNumbers:
    plotNumbers.append(maxLogLikelihoodTopicsNumber)

for numberOfTopics in plotNumbers:
    plt.figure()
    modelIdx = topicsRange.index(numberOfTopics)
    lda = ldaModels[modelIdx]
    sideLen = int(np.ceil(np.sqrt(numberOfTopics)))
    for topicIdx, topic in enumerate(lda.components_):
        ax = plt.subplot(sideLen, sideLen, topicIdx + 1)
        ax.imshow(topic.reshape((8, 8)), cmap=plt.cm.gray_r)
        untick(ax)
    plt.show()
