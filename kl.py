#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:48:40 2019

@author: Koryakovskiy Ivan (i.koryakovskiy@gmail.com)
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 1)

n = 100
eps = 1e-20

da = scipy.stats.norm(1, 3)
db = scipy.stats.norm(10, 3)

xa = da.rvs(n)
pa = da.pdf(xa) + eps
pb = db.pdf(xa) + eps

kl = np.mean(pa*np.log(pa/pb))
print("KL: " + str(kl))

axis.hist(xa, density=True, histtype='stepfilled', alpha=0.5)
plt.show()
