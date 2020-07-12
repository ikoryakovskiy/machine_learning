#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:42 2020

@author: ivan

https://medium.com/@hamza.emra/reinforcement-learning-with-tensorflow-2-0-cca33fead626
https://github.com/mrahtz/tensorflow-rl-pong/blob/master/pong.py
https://bair.berkeley.edu/blog/2019/10/14/functional-rl/
"""
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_policy_network(observations, actions, nh=256):
    policy = keras.Sequential()
    policy.add(layers.Dense(nh, activation='relu', input_shape=(observations,)))
    policy.add(layers.Dense(nh, activation='relu'))
    policy.add(layers.Dense(actions, activation='softmax'))
    return policy


def prepro_state(s):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    s = s[35:195] # crop
    s = s[::2,::2,0] # downsample by factor of 2
    s[s == 144] = 0 # erase background (background type 1)
    s[s == 109] = 0 # erase background (background type 2)
    s[s != 0] = 1 # everything else (paddles, ball) just set to 1
    s = s.astype(np.float).reshape((1, -1))
    return s


def discount_rewards(r, gamma=0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#@tf.function
def get_action_gradient(policy, s, loss):
    with tf.GradientTape() as tape:
        logits = policy(s)
        action_dist = logits.numpy()
        # Choose random action with p = action dist
        action = np.random.choice(action_dist[0], p = action_dist[0])
        action = np.argmax(action_dist == action)
        lo = loss(action, logits)
    grads = tape.gradient(lo, policy.trainable_variables) 
    return action + 2, grads

env = gym.make('Pong-v0')

actions = 2  # 3 == down, 2 == up
observations = 80 * 80

policy = create_policy_network(observations, actions)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

gamma = 0.8
update_every = 1

render = True

grad_buffer = policy.trainable_variables
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0

for episode in range(10000):
    
    cum_reward = 0
    img = prepro_state(env.reset())
    prev_img = np.zeros_like(img)
    ep_memory = []
    done = False
    while not done:
        if render: 
            env.render()
        
        # the state is the difference of two positions
        # positive pixel is current position, negative pixel is previous position
        state = img - prev_img
        prev_img = img
        action, grads = get_action_gradient(policy, state, loss)
        img, r, done, _ = env.step(action)
        
        img = prepro_state(img)
        cum_reward += r
        ep_memory.append([grads, r])
    
        if r:
            # Discound the rewards (esp useful to propagate rewards)
            ep_memory = np.array(ep_memory)
            ep_memory[:, 1] = discount_rewards(ep_memory[:, 1], gamma)
            
            for grads, r in ep_memory:
                for ix, grad in enumerate(grads):
                    grad_buffer[ix] += grad * r
            
            if episode % update_every == 0:
                optimizer.apply_gradients(zip(grad_buffer, policy.trainable_variables))
                for ix,grad in enumerate(grad_buffer):
                    grad_buffer[ix] = grad * 0
            ep_memory = []
    
    
    print(f"Episode {episode}, return {cum_reward}")
    print(policy.get_weights()[-1])


env.close()