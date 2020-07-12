#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:48:40 2019

@author: Koryakovskiy Ivan (i.koryakovskiy@gmail.com)
"""

import numpy as np

def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal gaussian distribution.

    :param log_std: Log of the standard deviation
    """
    return np.sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


# Compare
sigma = 10
# vs
#sigma = 1

mu = 0

n = 100
bounds = [-20, 20]

arr = np.linspace(bounds[0], bounds[1], n);
p = np.exp( - (arr - mu)**2 / (2*sigma**2))
p = p / np.sum(p)

# entropy
data_driven_H = -np.sum(p*np.log(p))
closed_form_H = gaussian_entropy(np.log(sigma))

#print(p)
print("Data-driven entropy: " + str(data_driven_H))
print("Closed-form entropy: " + str(closed_form_H))
