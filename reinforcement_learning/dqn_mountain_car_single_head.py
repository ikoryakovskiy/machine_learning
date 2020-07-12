#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:35:42 2020

@author: ivan

DQN algorithm for a discrete MountainCar

Observation:
    Type: Box(2)
    Num    Observation               Min            Max
    0      Car Position              -1.2           0.6
    1      Car Velocity              -0.07          0.07
Actions:
    Type: Discrete(3)
    Num    Action
    0      Accelerate to the Left
    1      Don't accelerate
    2      Accelerate to the Right
    Note: This does not affect the amount of velocity affected by the
    gravitational pull acting on the car.
Reward:
     Reward of 0 is awarded if the agent reached the flag (position = 0.5)
     on top of the mountain.
     Reward of -1 is awarded if the position of the agent is less than 0.5.
Starting State:
     The position of the car is assigned a uniform random value in
     [-0.6 , -0.4].
     The starting velocity of the car is always assigned to 0.
 Episode Termination:
     The car position is more than 0.5
     Episode length is greater than 200

"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class PlotFunction():
    def __init__(self, lims, n=100):
        xx, yy = np.meshgrid(
            np.linspace(lims[0], lims[1], n), np.linspace(lims[2], lims[3], n)
        )
        self.points = np.vstack([xx.flatten(), yy.flatten()]).T
        self.lims = lims
        self.n = n
        self.extent = lims

    def plot(self, imgs, titles, trajectory=None, zlim=None, states=None):
        n = len(imgs)
        fig, axs = plt.subplots(nrows=n)
        if not zlim:
            zlim = [None] * n
        for ax, img, title, lim in zip(axs, imgs, titles, zlim):
            img = np.reshape(img, newshape=(self.n, self.n)).transpose()  # order (rows, cols) required by imshow
            if not lim:
                vmin, vmax = img.min(), img.max()
            else:
                vmin, vmax = lim[0], lim[1]
            pos = ax.imshow(
                img, vmin=vmin, vmax=vmax,
                interpolation='none', extent=self.extent, aspect='auto', cmap='bwr'
            )

            if trajectory is not None:
                ax.quiver(
                    trajectory[:-1, 0],
                    trajectory[:-1, 1],
                    trajectory[1:, 0] - trajectory[:-1, 0],
                    trajectory[1:, 1] - trajectory[:-1, 1]
                )

            if states is not None:
                ax.scatter(states[:, 0], states[:, 1], marker="+")

            ax.set_xlim(self.lims[0], self.lims[1])
            ax.set_ylim(self.lims[2], self.lims[3])
            ax.set_title(title)
            fig.colorbar(pos, ax=ax)
        plt.show()


class ReplayMemory:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.write_pos = 0
        width = 2 * obs_dim + action_dim + 1 + 1
        self.memory = np.zeros((capacity, width))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0

    def push(self, state, action, next_state, reward, absorbing):
        transition = np.concatenate((state, action, next_state, np.array([reward]), np.array([absorbing])))
        self.memory[self.write_pos] = transition
        self.write_pos += 1
        if self.write_pos >= self.capacity:
            self.write_pos = 0
            self.size = self.capacity

    def sample(self, batch_size):
        size = max(self.size, self.write_pos)
        idxs = np.random.choice(size, batch_size, replace=False)
        batch = self.memory[idxs]
        return batch[:, :self.obs_dim], \
            batch[:, self.obs_dim:self.obs_dim + self.action_dim], \
            batch[:, self.obs_dim + self.action_dim: 2 * self.obs_dim + self.action_dim], \
            batch[:, 2 * self.obs_dim + self.action_dim: 2 * self.obs_dim + self.action_dim + 1], \
            batch[:, 2 * self.obs_dim + self.action_dim + 1: 2 * self.obs_dim + self.action_dim + 2]

    def __len__(self):
        return len(self.memory)


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


class ActionSelection:
    def __init__(self, action_bounds, max_steps):
        self.action_bounds = action_bounds
        self.max_steps = max_steps

    def reset(self, sigma=0.5, tau=10):
        self.noise = self.generate_ou(0, sigma, tau)
        action = np.clip(np.round(5*self.noise), self.action_bounds[0], self.action_bounds[1])
        self.action = action.astype(int)
        self.ss = 0

    def generate_ou(self, mu, sigma, tau):
        dt = .001  # Time step.
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)
        x = np.zeros(self.max_steps)
        # x[0] = np.random.uniform(mu - 0.2, mu + 0.2)
        for i in range(self.max_steps - 1):
            x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()
        return x

    def plot_noise(self):
        plt.plot(self.action)
        plt.show()


def center_action(action):
    return action - 1


def uncenter_action(action):
    return action + 1


def plot_data(function_plotter, q_function, centered_actions, traj=None, states=None):
    action_num = centered_actions.size
    num_points = function_plotter.points.shape[0]
    plot_states = np.tile(function_plotter.points, (action_num, 1))
    plot_actions = np.repeat(centered_actions, num_points, axis=0)[:, None]
    plot_state_actions = np.hstack((plot_states, plot_actions))
    qq = q_function(plot_state_actions).numpy()
    qq = qq.reshape((action_num, num_points)).T
    max_a = np.argmax(qq, axis=1)
    max_q = np.choose(max_a, qq.T).reshape((-1, 1))
    max_a_centered = centered_actions[max_a]
    zlim = [[-250, 50], [min(centered_actions), max(centered_actions)]]
    function_plotter.plot([max_q, max_a_centered], ["max_q", "policy"], trajectory=traj, zlim=zlim, states=states)
    normalizer.print_mean_std()


n_episodes = 2000
scores = []
render = True
gamma = 0.99
eps = 1.0
batch_size = 128
ou_sigma = 5

env = gym.make('MountainCar-v0')
# env = gym.make('Pendulum-v0')

obs_dim = env.observation_space.shape[0]
action_dim = 1
action_num = env.action_space.n

tf_opt = tf.keras.optimizers.Adam
# tf_opt = tf.keras.optimizers.RMSprop
hl = 64
nn_reg = tf.keras.regularizers.l2(0.01)
q_function = tf.keras.Sequential()
q_function.add(tf.keras.layers.Dense(hl, input_dim=obs_dim + action_dim, activation='relu', kernel_regularizer=nn_reg))
q_function.add(tf.keras.layers.Dense(hl, activation='relu', kernel_regularizer=nn_reg))
q_function.add(tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=nn_reg))
q_function.build()
q_function_optimizer = tf_opt(learning_rate=0.001)
#q_function_loss = tf.keras.losses.MeanSquaredError()
q_function_loss = tf.keras.losses.Huber()

valid_actions = [0, 1, 2]
centered_actions = np.fromiter(map(center_action, valid_actions), dtype=int)

action_selection_strategy = ActionSelection(
    action_bounds=(min(centered_actions), max(centered_actions)), max_steps=200
)
action_selection_strategy.reset(ou_sigma, tau=10)
action_selection_strategy.plot_noise()

memory = ReplayMemory(50000, obs_dim, action_dim)
normalizer = Normalize(obs_dim)
function_plotter = PlotFunction([-2.5, 2.5, -0.7, 0.7])

state_action = np.zeros((1, obs_dim + action_dim), dtype=np.float32)

plot_data(function_plotter, q_function, centered_actions)

#good_policy = np.concatenate((-np.ones(20), np.ones(40), -np.ones(60), np.ones(100)))
#good_policy = good_policy.astype(int)

for episode in range(n_episodes):
    state = env.reset()
    state = state.astype(np.float64)
    normalizer.update(state)
    normalized_state = normalizer.normalize(state)

    action_selection_strategy.reset(ou_sigma)

    ep_score = 0
    done = False
    grads = None

    ss = 0

    eps = max(0.05, eps * 0.998)

    trajectory = []
    while not done:

        if render:
            env.render()
        trajectory.append(normalized_state)

        if np.random.uniform() > eps:
            qq = []
            state_action[:, :obs_dim] = normalized_state.astype(np.float32)
            for ca in centered_actions:
                state_action[:, obs_dim:] = ca
                qq.append(q_function(state_action).numpy().flatten()[0])
            action = np.argmax(qq)
            centered_action = center_action(action)
        else:
            #centered_action = good_policy[ss]
            centered_action = action_selection_strategy.action[ss]
            #centered_action = np.random.choice(centered_actions)
            action = uncenter_action(centered_action)

        # take the choosen action
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float64)
        normalizer.update(next_state)
        normalized_next_state = normalizer.normalize(next_state)

        # prepare next iteration
        if done and not info.get('TimeLimit.truncated'):
            print("Doing good!===============================================")
        ep_score += reward

        absorbing = 1 if done and not info.get('TimeLimit.truncated') else 0
        memory.push(
            normalized_state.flatten(), np.atleast_1d(centered_action), normalized_next_state.flatten(), reward, absorbing
        )

        ss += 1
        normalized_state = normalized_next_state

    scores.append(ep_score)

    # update q_function
    batch_state, batch_action, batch_next_state, batch_reward, batch_absorbing = \
        memory.sample(batch_size)

    # calculate target values
    batch_next_state = np.tile(batch_next_state, (action_num, 1))
    batch_next_action = np.repeat(centered_actions, batch_size, axis=0)[:, None]
    next_state_actions = np.hstack((batch_next_state, batch_next_action))
    next_qq = q_function(next_state_actions).numpy()
    next_qq = next_qq.reshape((action_num, batch_size)).T
    max_a = np.argmax(next_qq, axis=1)
    max_q = np.choose(max_a, next_qq.T).reshape((-1, 1))
    target = batch_reward + (1 - batch_absorbing) * gamma * max_q

    state_actions = np.hstack((batch_state, batch_action))
    with tf.GradientTape() as tape:
        qq = q_function(state_actions)
        loss = q_function_loss(qq, target)
    grads = tape.gradient(loss, q_function.trainable_variables)

    q_function_optimizer.apply_gradients(zip(grads, q_function.trainable_variables))

    print(f"""Episode  {episode},   Score  {np.mean(scores[-100:]):.2f},   Epsilon {eps:.2f}""")

    if episode % 10 == 0:
        traj = np.reshape(trajectory, (-1, 2))
        plot_data(function_plotter, q_function, centered_actions, traj, states=batch_state)
        #action_selection_strategy.plot_noise()
