#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:42 2020

@author: ivan
"""

import gym
import numpy as np


env = gym.make('FrozenLake-v0', is_slippery=False)

s0 = env.reset()
qf = np.random.random(size=(4 * 4, 4)) - 0.5

epsilon = 0.05
alpha = 0.01
gamma = 0.97

cum_reward = 0
for it in range(100000):
    if np.random.random() < epsilon:
        a = np.random.randint(0, 4)
    else:
        a = np.argmax(qf[s0])
    s1, r, done, _ = env.step(a)

    done = int(done)
    cum_reward += r

    td = r + gamma * max(qf[s1]) * (1 - done) - qf[s0, a]
    qf[s0, a] += alpha * td

    s0 = s1
    if done:
        print(cum_reward)
        cum_reward = 0
        s0 = env.reset()

done = 0
cum_reward = 0
s = env.reset()
while done == 0:
    a = np.argmax(qf[s])
    s, r, done, _ = env.step(a)
    cum_reward += r

print(f"Final return {cum_reward}")
