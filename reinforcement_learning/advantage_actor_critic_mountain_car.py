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

Great courses:
    https://github.com/yandexdataschool/Practical_RL
    https://github.com/bayesgroup/deepbayes-2018
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# TASK = 1: REINFORCE    (learns fine using cumulative reward)
# TASK = 2: A2C          (does not learn becasue high variance and no cumulative rerards)
# TASK = 3: A2C + GAE    (learns well up to return 90, but then overshoots and forgets for some reason)
TASK = 3


class PlotFunction():
    def __init__(self, lims, n=100):
        xx, yy = np.meshgrid(
            np.linspace(lims[0], lims[2], n), np.linspace(lims[1], lims[3], n)
        )
        self.points = np.vstack([xx.flatten(), yy.flatten()]).T
        self.lims = lims
        self.n = n
        self.extent = [lims[0], lims[2], lims[1], lims[3]]

    def plot(self, function, title, trajectory=None, activation=None, zlim=None):
        z = function(self.points).numpy()
        z = z[:, 0]  # take first value which is mean value in case of policy
        if activation:
            z = activation(z)
        img = np.reshape(z, newshape=(self.n, self.n)).transpose()  # order (rows, cols) required by imshow
        if zlim:
            vmin, vmax = zlim
        else:
            vmin, vmax = z.min(), z.max()
        fig, ax = plt.subplots(ncols=1)
        pos = ax.imshow(img, vmin=vmin, vmax=vmax, interpolation='none', extent=self.extent, aspect='auto', cmap='bwr')

        if trajectory is not None:
            ax.quiver(
                trajectory[:-1, 0],
                trajectory[:-1, 1],
                trajectory[1:, 0] - trajectory[:-1, 0],
                trajectory[1:, 1] - trajectory[:-1, 1]
            )

        ax.set_xlim(self.lims[0], self.lims[2])
        ax.set_ylim(self.lims[1], self.lims[3])
        ax.set_title(title)
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


def target_train(target, ori, tau):
    weights = ori.get_weights()
    target_weights = target.get_weights()
    for ii in range(len(target_weights)):
        target_weights[ii] = tau * weights[ii] + (1 - tau) * target_weights[ii]
    target.set_weights(target_weights)


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
beta = 1e-4
ou_sigma = 1
tau = 0.001
lmbd = 0.999

entropy_c = np.sqrt(2 * np.pi * np.e)

goal_position = 0.45

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
tf_opt = tf.keras.optimizers.Adam
# tf_opt = tf.keras.optimizers.RMSprop
hl = 64
policy = tf.keras.Sequential()
policy_reg = tf.keras.regularizers.l2(0.0001)
policy.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='relu', kernel_regularizer=policy_reg))
policy.add(tf.keras.layers.Dense(hl, activation='relu', kernel_regularizer=policy_reg))
policy.add(tf.keras.layers.Dense(2, activation="linear", kernel_regularizer=policy_reg))
policy.build()
policy_optimizer = tf_opt(learning_rate=0.003)

value = tf.keras.Sequential()
near_zero = 'glorot_uniform'  # tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
value.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='relu',
                                kernel_initializer=near_zero, kernel_regularizer=policy_reg))
value.add(tf.keras.layers.Dense(hl, activation='relu', kernel_initializer=near_zero, kernel_regularizer=policy_reg))
value.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer=near_zero, kernel_regularizer=policy_reg))
value.build()
value_optimizer = tf_opt(learning_rate=0.003)
value_loss = tf.keras.losses.MeanSquaredError()

target_value = tf.keras.Sequential()
target_value.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='relu'))
target_value.add(tf.keras.layers.Dense(hl, activation='relu'))
target_value.add(tf.keras.layers.Dense(1, activation="linear"))
target_value.build()

# copy weights
target_value.set_weights(value.get_weights())

normalizer = Normalize(obs_dim)

anal = Analytics()
function_plotter = PlotFunction([-2.5, -0.7, 2.5, 0.7])

gradBuffer = policy.trainable_variables
for ix, grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0

value_grads_buffer = value.trainable_variables
for ix, grad in enumerate(value_grads_buffer):
    value_grads_buffer[ix] = grad * 0

for episode in range(n_episodes):
    state = env.reset()
    state = state.astype(np.float64)
    normalizer.update(state)
    normalized_state = normalizer.normalize(state)
    ep_memory = [[None, 0, normalized_state, 0]]

    action_selection_strategy.reset(ou_sigma)
    entropy_mem = []
    ep_score = 0
    done = False
    grads = None

    ss = 0

    trajectory = []
    while not done:

        if render:
            env.render()
        trajectory.append(normalized_state)

        with tf.GradientTape() as tape:
            # Forward pass
            mean_std = policy(normalized_state.astype(np.float32))
            mean, std = tf.split(mean_std, 2, axis=-1)
            tmean = tf.tanh(mean)
            tstd = tf.sigmoid(std)
            action = action_selection_strategy.sample(tmean.numpy(), tstd.numpy(), False)
            # Remove constant: - 0.5 * np.log(2 * np.pi)
            logpi = - tf.math.log(tstd) - 0.5 / (tstd ** 2) * (action - tmean) ** 2
            entropy = tf.math.log(tstd * entropy_c)
            loss = - logpi - beta * entropy

        grads = tape.gradient(loss, policy.trainable_variables)

        # take the choosen action
        state, reward, done, info = env.step(action)
        state = state.T.astype(np.float64)
        normalizer.update(state)
        normalized_state = normalizer.normalize(state)

        # prepare next iteration
        if done and not info.get('TimeLimit.truncated'):
            print("Doing good!")
        ep_score += reward
        absorbing = 1 if done and not info.get('TimeLimit.truncated') else 0
        ep_memory.append([grads, reward, normalized_state, absorbing])
        entropy_mem.append(entropy)
        anal.append([tf.reduce_mean(mean).numpy(), tf.reduce_mean(std).numpy()])
        ss += 1

    scores.append(ep_score)

    ep_memory = np.array(ep_memory)

    # Discound the rewards
    if TASK == 1:
        # Note: we discount rewards only in case when value function is not used
        ep_memory[:, 1] = discount_rewards(ep_memory[:, 1], gamma)

    # update value function
    advantage = np.zeros(ep_memory.shape[0])
    mean_value_loss = 0
    for ii in range(1, ep_memory.shape[0]):
        state = ep_memory[ii - 1][2]
        _, reward, next_state, absorbing = ep_memory[ii]
        # Target should be more stable and contain a bit more information.
        # This can be seen by the fact that we use target_value network and reward
        v_next = target_value(next_state).numpy()
        target = reward + (1 - absorbing) * gamma * v_next
        with tf.GradientTape() as tape:
            v_curr = value(state)
            value_lo = value_loss(v_curr, target)
        if TASK == 1:
            advantage[ii] = reward
        elif TASK >= 2:
            # For advantage estimation it is very important to use the slow-changing target function.
            # Otherwise, there will be a high variance and bias due to the quickly changing backuped value function
            v_curr_from_tgt = target_value(state).numpy()
            # Note: multiplying by 0 will result in REINFORCE
            advantage[ii] = reward + (gamma * v_next - v_curr_from_tgt) * 1.0
        value_grads = tape.gradient(value_lo, value.trainable_variables)
        mean_value_loss += value_lo.numpy()
        for jj, grad in enumerate(value_grads):
            value_grads_buffer[jj] += grad

    # optimize value function and the target value function
    value_optimizer.apply_gradients(zip(value_grads_buffer, value.trainable_variables))
    target_train(target_value, value, tau)

    mean_value_loss /= (ep_memory.shape[0] - 1)

    for jj, grad in enumerate(value_grads_buffer):
        value_grads_buffer[jj] = grad * 0

    # calculate generalized advantage. Note it's similarity with discount_rewards function!!!
    if TASK == 3:
        for ii in reversed(range(ep_memory.shape[0] - 1)):  # - 2, -1, -1):
            advantage[ii] += gamma * lmbd * advantage[ii + 1]

    # update policy
    for ii in range(1, ep_memory.shape[0]):
        policy_grads = ep_memory[ii][0]
        for jj, grad in enumerate(policy_grads):
            gradBuffer[jj] += grad * advantage[ii]

    if episode % update_every == 0:
        policy_optimizer.apply_gradients(zip(gradBuffer, policy.trainable_variables))
        for jj, grad in enumerate(gradBuffer):
            gradBuffer[jj] = grad * 0

    print(f"""Episode  {episode},   Score  {np.mean(scores[-100:]):.2f},   """
          f"""Entropy  {np.mean(entropy_mem):.2f},   Value loss {mean_value_loss:.5f}""")

    if episode % 10 == 0:
        anal.histogram()
        anal.reset()

        traj = np.reshape(trajectory, (-1, 2))
        function_plotter.plot(policy, "policy", traj, activation=np.tanh, zlim=[-1, 1])
        function_plotter.plot(target_value, "target_value", traj)

        normalizer.print_mean_std()

