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
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer


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
                ax.plot(trajectory[:, 0], trajectory[:, 1], color='cyan')

            if states is not None:
                ax.scatter(states[:, 0], states[:, 1], marker="+")

            ax.set_xlim(self.lims[0], self.lims[1])
            ax.set_ylim(self.lims[2], self.lims[3])
            ax.set_title(title)
            fig.colorbar(pos, ax=ax)
        plt.show()


class Normalize:
    def __init__(self, obs_dim):
        self.mean = np.zeros((1, obs_dim), dtype=np.float32)
        self.sqsum = np.zeros((1, obs_dim), dtype=np.float32)
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
        var = np.maximum([0.0001, 0.0001], self.sqsum / self.n)
        return (state - self.mean) / np.sqrt(var)

    def print_mean_std(self):
        if self.n > 0:
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
    qq = q_function(function_plotter.points).numpy()
    max_a = np.argmax(qq, axis=1)
    max_q = np.choose(max_a, qq.T).reshape((-1, 1))
    max_a_centered = centered_actions[max_a]
    #zlim = [[-250, 50], [min(centered_actions), max(centered_actions)]]
    zlim = [None, [min(centered_actions), max(centered_actions)]]
    function_plotter.plot([max_q, max_a_centered], ["max_q", "policy"], trajectory=traj, zlim=zlim, states=states)
    normalizer.print_mean_std()


def target_train(target, ori, tau):
    weights = ori.get_weights()
    target_weights = target.get_weights()
    for ii in range(len(target_weights)):
        target_weights[ii] = tau * weights[ii] + (1 - tau) * target_weights[ii]
    target.set_weights(target_weights)


def calc_epsilon(eps_initial, eps_final, steps_done, max_steps, replay_buffer_start_size,
                 eps_annealing_steps, eps_final_frame=0.01, evaluation=False, eps_evaluation=0.0):
    """Get the appropriate epsilon value from a given frame number
    Arguments:
        frame_number: Global frame number (used for epsilon)
        evaluation: True if the model is evaluating, False otherwise
        (uses eps_evaluation instead of default epsilon value)
    Returns:
        The appropriate epsilon value
    """
    # Slopes and intercepts for exploration decrease
    # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
    slope = -(eps_initial - eps_final) / eps_annealing_steps
    intercept = eps_initial - slope * replay_buffer_start_size
    slope_2 = -(eps_final - eps_final_frame) / (max_steps - eps_annealing_steps - replay_buffer_start_size)
    intercept_2 = eps_final_frame - slope_2 * max_steps

    if evaluation:
        return eps_evaluation
    elif steps_done < replay_buffer_start_size:
        return eps_initial
    elif steps_done >= replay_buffer_start_size and steps_done < replay_buffer_start_size + eps_annealing_steps:
        return slope * steps_done + intercept
    elif steps_done >= replay_buffer_start_size + eps_annealing_steps:
        return slope_2 * steps_done + intercept_2


scores = []
render = True
gamma = 0.97
eps = 1.0
batch_size = 32
ou_sigma = 5
flag_position = 0.5

# PER
use_per = True
max_steps = 1000000 # 25000000
replay_buffer_start_size = 50000 #50000
eps_annealing_steps = 500000 # 1000000
buffer_size = max_steps #1000000
eps_initial = 1.0
eps_final = 0.1
#

env = gym.make('MountainCar-v0')
# env = gym.make('Pendulum-v0')

obs_dim = env.observation_space.shape[0]
action_dim = 1
action_num = env.action_space.n

tf_opt = tf.keras.optimizers.Adam
# tf_opt = tf.keras.optimizers.RMSprop
hl = 64
dropout_rate = 0.5
nn_reg = tf.keras.regularizers.l2(0.1)
q_function = tf.keras.Sequential()
q_function.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='relu', kernel_regularizer=nn_reg))
q_function.add(tf.keras.layers.LayerNormalization())
q_function.add(tf.keras.layers.Dropout(dropout_rate))
q_function.add(tf.keras.layers.Dense(hl, activation='relu', kernel_regularizer=nn_reg))
q_function.add(tf.keras.layers.LayerNormalization())
q_function.add(tf.keras.layers.Dropout(dropout_rate))
q_function.add(tf.keras.layers.Dense(action_num, activation="linear", kernel_regularizer=nn_reg))
q_function.build()
q_function_optimizer = tf_opt(learning_rate=0.001)
# q_function_loss = tf.keras.losses.MeanSquaredError()
q_function_loss = tf.keras.losses.Huber()

q_target = tf.keras.Sequential()
q_target.add(tf.keras.layers.Dense(hl, input_dim=obs_dim, activation='relu', kernel_regularizer=nn_reg))
q_target.add(tf.keras.layers.LayerNormalization())
q_function.add(tf.keras.layers.Dropout(dropout_rate))
q_target.add(tf.keras.layers.Dense(hl, activation='relu', kernel_regularizer=nn_reg))
q_target.add(tf.keras.layers.LayerNormalization())
q_function.add(tf.keras.layers.Dropout(dropout_rate))
q_target.add(tf.keras.layers.Dense(action_num, activation="linear", kernel_regularizer=nn_reg))
q_target.build()

# copy weights
q_target.set_weights(q_function.get_weights())

valid_actions = [0, 1, 2]
centered_actions = np.fromiter(map(center_action, valid_actions), dtype=int)

action_selection_strategy = ActionSelection(
    action_bounds=(min(centered_actions), max(centered_actions)), max_steps=200
)
action_selection_strategy.reset(ou_sigma, tau=10)
action_selection_strategy.plot_noise()

if use_per:
    memory = PrioritizedReplayBuffer(buffer_size, obs_dim, action_dim)
else:
    memory = ReplayBuffer(buffer_size, obs_dim, action_dim)


normalizer = Normalize(obs_dim)

function_plotter = PlotFunction([-4, 4, -4, 4])

good_policy = np.concatenate((-np.ones(20), np.ones(40), -np.ones(60), np.ones(100))).astype(int)

plot_period = 10
steps_done = 0
episode = -1
while steps_done < max_steps:
    episode += 1
    state = env.reset()
    state = state.astype(np.float32)

    action_selection_strategy.reset(ou_sigma)

    ep_score = 0
    done = False
    grads = None

    ss = 0

    # eps = max(0.05, eps * 0.998)
    # eps = 0.05 + (0.9 - 0.05) * np.exp(-1. * steps_done / 40000)

    trajectory = []
    while not done:

        if render:
            env.render()

        normalizer.update(state)
        normalized_state = normalizer.normalize(state)

        if episode % plot_period == 0:
            trajectory.append(state)

        # Calculate epsilon based on the frame number
        eps = calc_epsilon(eps_initial, eps_final, steps_done, max_steps, replay_buffer_start_size,
                           eps_annealing_steps)

        if np.random.uniform() > eps:
            qq = q_function(normalized_state).numpy().flatten()
            action = np.argmax(qq)
            centered_action = center_action(action)
        else:
            if eps < 0.1:
                centered_action = action_selection_strategy.action[ss]
            else:
                centered_action = good_policy[ss]
            #centered_action = action_selection_strategy.action[ss]
            action = uncenter_action(centered_action)

        # take the choosen action
        next_state, reward, done, info = env.step(action)
        next_state = next_state.astype(np.float32)

        if done and not info.get('TimeLimit.truncated'):
            reward = 100  # there is a bug in MountainCar-v0
            print(f"Doing good!=============================================== {reward}")

#        reward -= 0.01 * centered_action ** 2
#        reward -= 0.01 * (flag_position - next_state[0]) ** 2
        absorbing = 1 if done and not info.get('TimeLimit.truncated') else 0

        memory.push(
            state.flatten(), np.atleast_1d(centered_action), next_state.flatten(), reward, absorbing
        )

        # prepare next iteration
        ep_score += reward

        ss += 1

        state = next_state

    steps_done += ss
    scores.append(ep_score)

    if steps_done > replay_buffer_start_size:
        # update q_function
        if use_per:
            (batch_state, batch_action, batch_next_state, batch_reward, batch_absorbing), importance, indices = \
                memory.sample(batch_size, priority_scale=0.7)
            eps = calc_epsilon(eps_initial, eps_final, steps_done, max_steps, replay_buffer_start_size,
                               eps_annealing_steps)
            importance = importance ** (1 - eps)
        else:
            batch_state, batch_action, batch_next_state, batch_reward, batch_absorbing = \
                memory.sample(batch_size)

        if np.any(batch_reward >= 0):
            print("Sampled 0+++++++++++++++++++++++++++++++++++++++++++++++++")

        # normalize states
        batch_state = normalizer.normalize(batch_state)
        batch_next_state = normalizer.normalize(batch_next_state)

        # calculate target values
        use_ddqn = True
        if use_ddqn:
            # Double DQN update
            next_qq = q_function(batch_next_state).numpy()
            next_max_a = np.argmax(next_qq, axis=1)
            target_qq = q_target(batch_next_state).numpy()
        else:
            target_qq = q_target(batch_next_state).numpy()
            next_max_a = np.argmax(target_qq, axis=1)
        next_max_q = np.choose(next_max_a, target_qq.T).reshape((-1, 1))
        target = batch_reward + (1 - batch_absorbing) * gamma * next_max_q

        batch_action = uncenter_action(batch_action.flatten().astype(int))
        batch_action = tf.convert_to_tensor(batch_action)

        with tf.GradientTape() as tape:
            q_values = q_function(batch_state)
            # qq = tf.gather(q_values, batch_action, axis=1, batch_dims=1)[:, None]
            # losses should have same dimentionality!!

            # using tf.one_hot causes strange errors
            one_hot_actions = tf.keras.utils.to_categorical(batch_action, action_num, dtype=np.float32)
            qq = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)[:, None]

            error = qq - target
            loss = q_function_loss(qq, target)

            if use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

            grads = tape.gradient(loss, q_function.trainable_variables)

            q_function_optimizer.apply_gradients(zip(grads, q_function.trainable_variables))
            target_train(q_target, q_function, tau=0.001)

        if use_per:
            memory.set_priorities(indices, error)

        print(f"""Episode  {episode},   Score  {np.mean(scores[-100:]):.2f},   Epsilon {eps:.2f}""")

        if episode % plot_period == 0:
            traj = np.reshape(trajectory, (-1, 2))
            traj = normalizer.normalize(traj)
            plot_data(function_plotter, q_target, centered_actions, traj, states=batch_state)
            # action_selection_strategy.plot_noise()
