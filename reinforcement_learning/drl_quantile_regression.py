#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:44:36 2020

@author: ivan


Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/pdf/1710.10044.pdf
The solution is you got stuck you could cheat a little bit https://github.com/ars-ashuha/quantile-regression-dqn-pytorch
"""
from collections import deque
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from replay_buffers import ReplayBuffer

import random


def gather(theta_per_action, actions, batch_range):
    indices = tf.concat((batch_range, actions), axis=1)
    return tf.gather_nd(theta_per_action, indices)


# test = np.array([
#     [[0, 1], [2, 3]],
#     [[4, 5], [6, 7]],
# ])
# test = tf.constant(test)

# #indices = [[0, 1], [1, 0]]

# batch_size = 2
# batch_range = tf.reshape(tf.range(0, batch_size), (-1, 1))
# actions = tf.constant([[1], [0]])
# gathered = gather(test, actions, batch_range)

# print(gathered)
# print(gathered.shape)

# # [2, 3]
# # [4, 5]


def huber(x, k=1.0):
    absx = tf.math.abs(x)
    powx = tf.math.pow(x, 2)
    return tf.where(absx < k, 0.5 * powx, k * (absx - 0.5 * k))


class Network:
    def __init__(self, len_state, num_quant, num_actions):
        self.num_quant = num_quant
        self.num_actions = num_actions
        hl = 64
        nn_reg = tf.keras.regularizers.l2(0.1)
        ###########################################################
        # Define your model here, it is ok to use just
        # two layers and tanh nonlinearity, do not forget that
        # shape of the output should be
        # batch_size x self.num_actions x self.num_quant
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='tanh', kernel_regularizer=nn_reg))
        self.model.add(tf.keras.layers.Dense(hl, activation='tanh', kernel_regularizer=nn_reg))
        output_size = (num_actions, num_quant)
        self.model.add(tf.keras.layers.Dense(np.prod(output_size), activation="linear", kernel_regularizer=nn_reg))
        self.model.add(tf.keras.layers.Reshape(output_size))
        self.model.build()
        ###########################################################

    def forward(self, x):
        ###########################################################
        # Compute the output of the network
        if len(x.shape) == 1:
            x = x[None, :]
        return self.model(x)
        # Tensor of shape batch_size x self.num_actions x self.num_quant
        ###########################################################

    def select_action(self, state, eps):
        if random.random() > eps:
            ###########################################################
            action = self.calc_max_action(state)
            ###########################################################
        else:
            action = np.random.randint(0, self.num_actions)
        return int(action)

    def calc_max_action(self, state):
        action_quant = self.forward(state)
        qq = tf.reduce_mean(action_quant, axis=-1)
        action = tf.argmax(qq, axis=-1, output_type=tf.dtypes.int32)
        return action


def get_eps(steps, eps_start, eps_end, eps_dec):
    return eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)


def quantile_huber_loss(target, pred, actions, atoms, tau):
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
    pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, atoms])
    target_tile = tf.tile(tf.expand_dims(
        target, axis=1), [1, atoms, 1])
    huber_loss = huber_loss(target_tile, pred_tile)
    tau = tf.reshape(np.array(tau), [1, atoms])
    inv_tau = 1.0 - tau
    tau = tf.tile(tf.expand_dims(tau, axis=1), [1, atoms, 1])
    inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, atoms, 1])
    error_loss = tf.math.subtract(target_tile, pred_tile)
    loss = tf.where(tf.less(error_loss, 0.0), inv_tau * huber_loss, tau * huber_loss)
    loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(loss, axis=2), axis=1))
    return loss


# Params
num_quant = 51
render = True

# Here we've defined a schedule for exploration i.e. random action with prob eps
eps_start, eps_end, eps_dec = 0.9, 0.1, 500

env = gym.make('MountainCar-v0')
obs_dim = env.observation_space.shape[0]
action_dim = 1
action_num = env.action_space.n

memory = ReplayBuffer(10000, obs_dim, action_dim)

Z = Network(obs_dim, num_quant, action_num)
Ztgt = Network(obs_dim, num_quant, action_num)
Ztgt.model.set_weights(Z.model.get_weights())
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

tau = tf.constant(np.reshape((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant), (-1, 1, 1)), dtype=tf.float32)

gamma, batch_size = 0.99, 32
steps_done, running_reward = 0, 0
batch_range = tf.reshape(tf.range(0, batch_size), (-1, 1))

for episode in range(500):
    sum_reward = 0
    state = env.reset()
    while True:
        steps_done += 1
        if render:
            env.render()

        eps = get_eps(steps_done, eps_start, eps_end, eps_dec)
        action = Z.select_action(state, eps)
        next_state, reward, done, info = env.step(action)

        absorbing = 1 if done and not info.get('TimeLimit.truncated') else 0
        if done and not info.get('TimeLimit.truncated'):
            reward = 0  # there is a bug in MountainCar-v0

        memory.push(
            state.flatten(), np.atleast_1d(action), next_state.flatten(), reward, absorbing
        )
        sum_reward += reward
        state = next_state

        if len(memory) < batch_size:
            continue

        ###########################################################
        # Sample transitions from Replay Memory
        states, actions, next_states, rewards, dones = memory.sample(batch_size)
        ###########################################################

        ###########################################################
        # Calculate quantiles for the next stage with target network
        # and then take value for a max action
        next_theta_per_action = Ztgt.forward(states)
        max_action = Ztgt.calc_max_action(next_states)
        Znext_max = gather(next_theta_per_action, max_action[:, None], batch_range)
        Ttheta = rewards + gamma * (1 - dones) * Znext_max

        # ###
        # tau = np.arange(3).reshape((1, 3))
        # mul = np.arange(6).reshape((2, 3))
        # res = tau - np.expand_dims(mul, axis=0)
        # ###

        with tf.GradientTape() as tape:
            # Calculate quantiles theta for current state and actions
            theta_per_action = Z.forward(states)
            theta = gather(theta_per_action, actions, batch_range)

            # Calculate loss, use this trick to compute pairwise differences
            # Trick Tensor of shape (3,2,1) minus Tensor of shape (1,2,3) is Tensor of shape (3, 2, 3)
            # With all pairwise differences :)
            # Use Huber elementwise function to compute Huber loss
            # diff = tf.expand_dims(tf.transpose(Ttheta), axis=-1) - theta
            # loss = tf.reduce_mean(huber(diff) * tf.math.abs(tau - tf.cast(tf.stop_gradient(diff) < 0, tf.float32)))
            diff = Ttheta - theta
            hu = huber(diff)
            delta = tf.cast(tf.stop_gradient(diff) < 0, tf.float32)
            tau_multiplier = tf.math.abs(tau - np.expand_dims(delta, axis=0))
            loss = tf.reduce_mean(hu * tau_multiplier)
        ###########################################################

        grads = tape.gradient(loss, Z.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, Z.model.trainable_variables))

        taus = [(2*(i-1)+1)/(2*num_quant) for i in range(1, num_quant+1)]
        lossss = quantile_huber_loss(Ttheta, Z.forward(states), actions, num_quant, taus)

        if steps_done % 100 == 0:
            Ztgt.model.set_weights(Z.model.get_weights())

        if done and episode % 50 == 0:
            print(f"""Episode  {episode},   Steps  {steps_done:.2f},   Epsilon  {eps:.2f}""")
            print(f"""Return  {running_reward:.2f},   Loss {loss.numpy():.2f}""")

        if done:
            running_reward = sum_reward if not running_reward else 0.2 * sum_reward + running_reward * 0.8
            break
