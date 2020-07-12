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
a0 = np.argmax(qf[s0])
for it in range(100000):
    s1, r, done, _ = env.step(a0)

    done = int(done)
    cum_reward += r

    # select next action
    if np.random.random() < epsilon:
        a1 = np.random.randint(0, 4)
    else:
        a1 = np.argmax(qf[s1])

    td = r + gamma * qf[s1, a1] * (1 - done) - qf[s0, a0]
    qf[s0, a0] += alpha * td

    s0, a0 = s1, a1
    if done:
        print(cum_reward)
        cum_reward = 0
        s0 = env.reset()
        if np.random.random() < epsilon:
            a0 = np.random.randint(0, 4)
        else:
            a0 = np.argmax(qf[s0])

done = 0
cum_reward = 0
s = env.reset()
while done == 0:
    a = np.argmax(qf[s])
    s, r, done, _ = env.step(a)
    cum_reward += r

print(f"Final return {cum_reward}")
