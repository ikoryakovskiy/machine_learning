#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:42 2020

@author: ivan

Monte-Carlo Policy Gradient
Why we use cross-entropy loss? https://amoudgl.github.io/blog/policy-gradient/

All you need is R * H(y, a) where R is the discounted reward and H is the cross
entropy between your suggested action y and the actual action taken a.
For H you can use (Assuming action_taken is one-hot):
H = - tf.reduce_sum(action_taken * tf.log(action_probabilities), axis=1)

The final loss for t = 2 would be:
L_2 = reward * gamma * H(y_2, a_2)

At 300 episodes gets average return around 50

aka REINFORCE
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class PlotFunction():
    def __init__(self, lims, n=30):
        xx, yy = np.meshgrid(
            np.linspace(lims[0], lims[2], n), np.linspace(lims[1], lims[3], n)
        )
        self.points = np.vstack([xx.flatten(), yy.flatten()]).T
        self.lims = lims
        self.n = n
        self.extent = [lims[0], lims[2], lims[1], lims[3]]

    def plot(self, function, trajectory):
        z = function(self.points).numpy()
        z = z[:, 0]  # take first value which is mean value in case of policy
        img = np.reshape(z, newshape=(self.n, self.n)).transpose()
        fig, ax = plt.subplots(ncols=1)
        pos = ax.imshow(img, vmin=z.min(), vmax=z.max(), interpolation='none', extent=self.extent, aspect='auto')

        ax.quiver(
            trajectory[:-1, 0],
            trajectory[:-1, 1],
            trajectory[1:, 0] - trajectory[:-1, 0],
            trajectory[1:, 1] - trajectory[:-1, 1]
        )

        ax.set_xlim(self.lims[0], self.lims[2])
        ax.set_ylim(self.lims[1], self.lims[3])
        fig.colorbar(pos, ax=ax)
        plt.show()


class Analytics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []

    def append(self, value):
        self.values.append(value)

    def histogram(self):
        values = np.array(self.values)
        plt.hist(values, label=["mean", "std"])
        plt.legend()
        plt.show()
        minv = list(np.min(values, axis=0))
        maxv = list(np.max(values, axis=0))
        print(f"Mean, std stat {minv}, {maxv}")


class Normalize:
    def __init__(self, obs_dim):
        self.mean = np.zeros((1, obs_dim), dtype=np.float64)
        self.sqsum = np.zeros((1, obs_dim), dtype=np.float64)
        self.n = 0

    def update(self, state):
        prev_mean = self.mean.copy()
        self.n += 1
        self.mean = self.mean + (state - self.mean) / self.n
        self.sqsum = self.sqsum + (state - prev_mean) * (state - self.mean)

    def normalize(self, state):
        # also avoid calling when self.n == 0
        # should be var = self.sqsum / (self.n - 1), but for numerical
        # stability do it differently
        var = np.maximum([0.01, 0.01], self.sqsum / self.n)
        return (state - self.mean) / np.sqrt(var)

    def print_mean_std(self):
        std = np.sqrt(self.sqsum / self.n).flatten()
        print(f"State normalization mean {self.mean.flatten()} and std {std}")


def discount_rewards(rr, gamma=0.8):
    for t in reversed(range(rr.size - 1)):
        rr[t] = gamma * rr[t + 1] + rr[t]
    return rr


class ActionSelection:
    def __init__(self, action_bounds, max_steps):
        self.action_bounds = action_bounds
        self.max_steps = max_steps

    def reset(self, sigma=0.5, tau=10):
        self.noise = self.generate_ou(0, sigma, tau)
        self.ss = 0

    def generate_ou(self, mu, sigma, tau):
        dt = .001  # Time step.
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)
        x = np.zeros(self.max_steps)
        for i in range(self.max_steps - 1):
            x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()
        return x

    def plot_noise(self):
        plt.plot(self.noise)
        plt.show()

    def sample(self, mean, stddev, test):
        if test:
            action = mean
        else:
            action = mean + std * self.noise[self.ss]
            self.ss += 1
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])


n_episodes = 2000
scores = []
update_every = 1
render = True
gamma = 0.99
beta = 1e-1
ou_sigma = 1

entropy_c = np.sqrt(2 * np.pi * np.e)

goal_position = 0.45
test_period = 10

env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Pendulum-v0')

# obs_lims = np.concatenate((env.observation_space.low, env.observation_space.high))

obs_dim = env.observation_space.shape[0]
max_action = env.action_space.shape[0]
action_bounds = (env.action_space.low, env.action_space.high)
action_selection_strategy = ActionSelection(action_bounds, max_steps=1000)
action_selection_strategy.reset(ou_sigma)
action_selection_strategy.plot_noise()

# Note!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Although this is called a policy, in fact this is a function that gives parameters
# of the policy. The actual policy we are using is gaussian (or bernouli in case of discrete actions).
policy = tf.keras.Sequential()
policy_reg = tf.keras.regularizers.l2(0.01)
policy.add(tf.keras.layers.Dense(32, input_dim=obs_dim, activation='relu', kernel_regularizer=policy_reg))
policy.add(tf.keras.layers.Dense(2, activation="linear", kernel_regularizer=policy_reg))
policy.build()
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)


normalizer = Normalize(obs_dim)

anal = Analytics()
# function_plotter = PlotFunction(obs_lims)
function_plotter = PlotFunction([-2.5, -0.7, 2.5, 0.7])

gradBuffer = policy.trainable_variables
for ix, grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0

for episode in range(n_episodes):
    state = env.reset()
    action_selection_strategy.reset(ou_sigma)
    ep_memory = []
    entropy_mem = []
    ep_score = 0
    done = False

    ss = 0
    test = (episode % test_period == test_period - 1)
    if test:
        print("Testing")

    trajectory = []
    while not done:

        if render:
            env.render()

        with tf.GradientTape() as tape:
            # Forward pass
            normalizer.update(state)
            normalized_state = normalizer.normalize(state)
            trajectory.append(normalized_state)
            mean_std = policy(normalized_state.astype(np.float32))
            mean, std = tf.split(mean_std, 2, axis=-1)
            tmean = tf.tanh(mean)
            tstd = tf.sigmoid(std)
            action = action_selection_strategy.sample(tmean.numpy(), tstd.numpy(), test)
            # Remove constant from:
            # logpi = - 0.5 * np.log(2 * np.pi) - tf.math.log(stddev) - 0.5 / (stddev ** 2) * (action - mean) ** 2
            logpi = - tf.math.log(tstd) - 0.5 / (tstd ** 2) * (action - tmean) ** 2
            entropy = tf.math.log(tstd * entropy_c)
            loss = - logpi - beta * entropy

        if not test:
            grads = tape.gradient(loss, policy.trainable_variables)

        # take the choosen action
        state, reward, done, _ = env.step(action)
        state = state.flatten().astype(np.float64)

        # prepare next iteration
        if reward > 50:
            print("Doing good!")
        ep_score += reward
        ep_memory.append([grads, reward])
        entropy_mem.append(entropy)
        anal.append([tf.reduce_mean(mean).numpy(), tf.reduce_mean(std).numpy()])
        ss += 1

    scores.append(ep_score)
    # Discound the rewards
    ep_memory = np.array(ep_memory)
    ep_memory[:, 1] = discount_rewards(ep_memory[:, 1], gamma)

    for policy_grads, reward in ep_memory:
        for ix, grad in enumerate(policy_grads):
            gradBuffer[ix] += grad * reward

    if episode % update_every == 0:
        if not test:
            policy_optimizer.apply_gradients(zip(gradBuffer, policy.trainable_variables))
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

    print(f"""Episode  {episode},   Score  {np.mean(scores[-100:])},   """
          f"""Entropy  {np.mean(entropy_mem)}""")
    if test:
        anal.histogram()
        anal.reset()
        function_plotter.plot(policy, np.reshape(trajectory, (-1, 2)))
        normalizer.print_mean_std()
