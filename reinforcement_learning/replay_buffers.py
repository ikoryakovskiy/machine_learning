#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:12:03 2020

@author: ivan
"""


import os
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.write_pos = 0
        width = 2 * obs_dim + action_dim + 1 + 1
        self.memory = np.zeros((capacity, width), dtype=np.float32)
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
        if size >= batch_size:
            idxs = np.random.choice(size, batch_size, replace=False)
        else:
            idxs = np.random.choice(size, batch_size, replace=True)
        batch = self.memory[idxs]
        return batch[:, :self.obs_dim], \
            batch[:, self.obs_dim:self.obs_dim + self.action_dim], \
            batch[:, self.obs_dim + self.action_dim: 2 * self.obs_dim + self.action_dim], \
            batch[:, 2 * self.obs_dim + self.action_dim: 2 * self.obs_dim + self.action_dim + 1], \
            batch[:, 2 * self.obs_dim + self.action_dim + 1: 2 * self.obs_dim + self.action_dim + 2]

    def __len__(self):
        return max(self.size, self.write_pos)


class PrioritizedReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(self, capacity, obs_dim, action_dim, history_length=1, use_per=True):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.states = np.empty((self.size, self.obs_dim), dtype=np.float32)
        self.actions = np.empty((self.size, self.action_dim), dtype=np.int32)
        self.rewards = np.empty((self.size, 1), dtype=np.float32)
        self.absorbing_flags = np.empty((self.size, 1), dtype=np.int32)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def push(self, state, action, next_state, reward, absorbing, clip_reward=False):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if state.shape[0] != self.obs_dim:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.states[self.current, ...] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.absorbing_flags[self.current] = absorbing
        priority_offset = max(reward, 0)  # Ivan's modification for good rewards
        self.priorities[self.current] = max(self.priorities.max(), 1) + priority_offset  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def sample(self, batch_size=32, priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count - 1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based
                # on priority weights
                if self.use_per:
                    index = np.random.choice(np.arange(self.history_length, self.count - 1), p=sample_probabilities)
                else:
                    index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.
                # If either are True, the index is invalid.
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.absorbing_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.states[idx - self.history_length:idx, ...])
            new_states.append(self.states[idx - self.history_length + 1:idx + 1, ...])

        states = np.asarray(states).reshape((-1, self.obs_dim))
        new_states = np.asarray(new_states).reshape((-1, self.obs_dim))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1.0 / self.count * 1.0 / sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], new_states, self.rewards[indices], self.absorbing_flags[indices]), \
                importance, indices
        else:
            return states, self.actions[indices], new_states, self.rewards[indices], self.absorbing_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/states.npy', self.states)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/absorbing_flags.npy', self.absorbing_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.states = np.load(folder_name + '/states.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.absorbing_flags = np.load(folder_name + '/absorbing_flags.npy')