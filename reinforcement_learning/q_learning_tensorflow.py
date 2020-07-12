#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:42 2020

@author: ivan

Source:
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""

import gym
import numpy as np
import tensorflow as tf


model = tf.keras.Sequential()
inputs = tf.keras.Input(shape=(2,))
model.add(inputs)
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(1))
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
mse = tf.keras.losses.MeanSquaredError()


@tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        pred_y = model(inputs)
        loss_value = mse(target, pred_y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_action(model, s, epsilon):
    if np.random.random() < epsilon:
        a = np.random.randint(0, 4)
    else:
        q = []
        for a in range(4):
            x = np.array([[s0, a]])
            q.append(model.predict(x)[0, 0])
        a = np.argmax(q)
    return a


env = gym.make('FrozenLake-v0', is_slippery=False)

s0 = env.reset()
qf = np.random.random(size=(4 * 4, 4)) - 0.5

alpha = 0.01
epsilon = 0.05
gamma = 0.97

cum_reward = 0
for it in range(100000):
    a = get_action(model, s0, epsilon)
    s1, r, done, _ = env.step(a)

    done = int(done)
    cum_reward += r

    q1 = []
    for a1 in range(4):
        x = np.array([[s1, a1]])
        q1.append(model.predict(x)[0, 0])
    q1 = max(q1)
    q0 = model.predict(np.array([[s0, a]]).reshape((-1, 2)))[0, 0]
    td = r + gamma * q1 * (1 - done) - q0

    target = q0 + alpha * td

    loss, model_gradients = grad(model, np.array([s0, a]), target)

    sgd.apply_gradients(zip(model_gradients, model.trainable_variables))

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
