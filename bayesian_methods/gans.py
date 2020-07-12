#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:51:23 2020

@author: ivan

https://github.com/bayesgroup/deepbayes-2018/blob/master/day4_gans/GAN_deep_bayes_updated.ipynb
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl
import tensorflow.keras.optimizers as tko
import tensorflow.math as tm

import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

np.random.seed(12345)
plt.rcParams['figure.figsize'] = (12, 12)

# Task 1: Vanilla GAN
# Task 2: Generator loss - log G(D(z))
# Tase 3: Vanilla Wasserstein + weights clipping
# Tase 4: Vanilla Wasserstein + gradient penalty
# Task 5: Wasserstein Introspective Neural Networkss
TASK = 5


def sample_true(n, components, means, covs, comps_dist):
    comps = comps_dist.rvs(size=n)
    conds = np.arange(components)[:, None] == comps[None, :]
    arr = np.array([np.random.multivariate_normal(means[c], covs[c], size=n)
                     for c in range(components)])
    return np.select(conds[:, :, None], arr).astype(np.float32)


def sample_noise(n, noise_dim):
    return np.random.normal(size=(n, noise_dim)).astype(np.float32)


def vis_data(data, lims):
    """
        Visualizes data as histogram
    """
    hist = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)


def vis_g(generator, fixed_noise, lims):
    """
        Visualizes generator's samples as circles
    """
    if generator:
        data = generator(fixed_noise).numpy()
        if np.isnan(data).any():
            return

        plt.scatter(data[:, 0], data[:, 1], alpha=0.2, c='b')
        plt.xlim(lims)
        plt.ylim(lims)


def vis_points(data, lims, c='b'):
    """
        Visualizes the supplied samples as circles
    """
    if np.isnan(data).any():
        return

    plt.scatter(data[:, 0], data[:, 1], alpha=0.2, c=c)
    plt.xlim(lims)
    plt.ylim(lims)


def vis_d(generator, discriminator, fixed_noise, X_grid, Y_grid, grid, g_loss):
    """
        Visualizes discriminator's gradient on grid
    """
    # data_gen = generator(fixed_noise)
    # loss = d_loss(discriminator(data_gen), discriminator(grid))
    with tf.GradientTape() as tape:
        loss = g_loss(discriminator(grid))
    grads = -tape.gradient(loss, grid).numpy()
    plt.quiver(X_grid, Y_grid, grads[:, 0], grads[:, 1], color='black', alpha=0.9)


def get_grid(lims):
    X, Y = np.meshgrid(np.linspace(lims[0], lims[1], 30), np.linspace(lims[0], lims[1], 30))
    X = X.flatten()
    Y = Y.flatten()
    grid = tf.Variable(np.vstack([X, Y]).astype(np.float32).T)
    return X, Y, grid


def iterate_minibatches(X, batchsize, y=None):
    perm = np.random.permutation(X.shape[0])

    for start in range(0, X.shape[0], batchsize):
        end = min(start + batchsize, X.shape[0])
        if y is None:
            yield X[perm[start:end]]
        else:
            yield X[perm[start:end]], y[perm[start:end]]


def leaky_relu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)


def get_generator(noise_dim, out_dim, hidden_dim=100):
    if TASK == 5:
        return None
    layers = [
        tkl.Dense(hidden_dim, activation=leaky_relu, input_dim=noise_dim),
        tkl.Dense(hidden_dim, activation=leaky_relu),
        tkl.Dense(out_dim)
    ]
    return tk.Sequential(layers)


def get_discriminator(in_dim, hidden_dim=100):
    layers = [
        tkl.Dense(hidden_dim, activation=leaky_relu, input_dim=in_dim),
        tkl.Dense(hidden_dim, activation=leaky_relu),
        tkl.Dense(hidden_dim, activation=leaky_relu),
        tkl.Dense(1, activation='sigmoid'),
    ]
    return tk.Sequential(layers)


def g_loss(d_scores_fake):
    """
        `d_scores_fake` is the output of the discrimonator model applied to a batch of fake data

        NOTE: we always define objectives as if we were minimizing them (remember that maximize = negate and minimize)
    """
    if TASK == 1:
        return tm.reduce_mean(tm.log(1 - d_scores_fake))
    elif TASK == 2:
        return - tm.reduce_mean(tm.log(d_scores_fake))
    elif TASK == 3 or TASK == 4:
        # tries to maximize score such that is becomes positive
        # (similar to the discriminator score)
        return - tm.reduce_mean(d_scores_fake)
    elif TASK == 5:
        # INN does not generator
        return None


def d_loss(d_scores_fake, d_scores_real):
    """
        `d_scores_fake` is the output of the discrimonator model applied to a batch of fake data
        `d_scores_real` is the output of the discrimonator model applied to a batch of real data

        NOTE: we always define objectives as if we were minimizing them (remember that maximize = negate and minimize)
    """
    if TASK == 1:
        return - tm.reduce_mean(tm.log(d_scores_real) + tm.log(1 - d_scores_fake))
    elif TASK == 2:
        return - tm.reduce_mean(tm.log(d_scores_real) + tm.log(1 - d_scores_fake))
    elif TASK == 3 or TASK == 4:
        # Maximize Critic score
        # push real samples mean to large positive values,
        # and push fake scores mean to large negative values
        return - (tm.reduce_mean(d_scores_real) - tm.reduce_mean(d_scores_fake))
    elif TASK == 5:
        return - (tm.reduce_mean(d_scores_real) - tm.reduce_mean(d_scores_fake))


def inv_sigmoid(x):
    """ Computes the logit function, i.e. the sigmoid inverse. """
    return - tf.math.log(1. / x - 1.)


# params
noise_dim = 20
batch_size = 64
if TASK == 4 or TASK == 5:
    k_d, k_g = 5, 1  # IMPORTANT! Number of D updates per G update
else:
    k_d, k_g = 1, 1


# Model and optimizers
generator = get_generator(noise_dim, out_dim = 2)
discriminator = get_discriminator(in_dim = 2)

if TASK <= 2:
    lr = 0.001
    g_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
elif TASK == 3:
    lr = 0.001
    g_optimizer = tko.RMSprop(learning_rate=lr)
    d_optimizer = tko.RMSprop(learning_rate=lr)
elif TASK == 4:
    lr = 0.0002
    g_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
    d_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
elif TASK == 5:
    lr = 0.001
    g_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_optimizer = tko.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

# Generate data 2D data
lims = (-5, 5)
means = np.array([[-1, -3], [1, 3], [-2, 0]])
covs = np.array([[[1, 0.8], [0.8, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]])
probs = np.array([0.2, 0.5, 0.3])
components = len(means)

comps_dist = rv_discrete(values=(range(components), probs))

data = sample_true(100000, components, means, covs, comps_dist)

fixed_noise = tf.constant(sample_noise(1000, noise_dim))
X_grid, Y_grid, grid = get_grid(lims)

# For TASK 5
fake_batch = np.random.normal(size=(batch_size, data.shape[1]))

vis_data(data, lims)
if TASK < 5:
    vis_g(generator, fixed_noise, lims)
    vis_d(generator, discriminator, fixed_noise, X_grid, Y_grid, grid, g_loss)
else:
    vis_points(fake_batch, lims)
plt.show()


# Training
for it, real_data in enumerate(iterate_minibatches(data, batch_size)):

    real_data = tf.constant(real_data)

    # Optimize D
    for _ in range(k_d):

        if TASK < 5:
            # Sample noise
            noise = tf.constant(sample_noise(real_data.shape[0], noise_dim))
            fake_data = generator(noise)
        else:
            fake_idxs = np.random.choice(range(fake_batch.shape[0]), size=batch_size, replace=False)
            fake_data = fake_batch[fake_idxs]

        # Compute gradient
        with tf.GradientTape() as tape:
            loss = d_loss(discriminator(fake_data), discriminator(real_data))
            if TASK == 4 or TASK == 5:
                # Gradient Penalty
                uu = np.random.uniform(0, 1, size=(real_data.shape[0], 1))
                interpolated = uu * real_data + (1 - uu) * fake_data
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    critic_logits = discriminator(interpolated)
                d_critic = gp_tape.gradient(critic_logits, interpolated)
                d_critic += 1e-16
                gradient_penalty = tf.reduce_mean((tf.norm(d_critic, axis=1) - 1) ** 2)
                loss += 0.1 * gradient_penalty
        print(loss)

        grads = tape.gradient(loss, discriminator.trainable_variables)

        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        if TASK == 3:
            # clipping gradients can lead to exploding or vanishing gradients
            wgan_clip = 0.01
            for ix, tv in enumerate(discriminator.trainable_variables):
                clipped_w = tf.clip_by_value(tv, - wgan_clip, wgan_clip)
                tf.keras.backend.set_value(discriminator.trainable_variables[ix], clipped_w)

    # Optimize G
    for _ in range(k_g):
        if TASK < 5:
            # Sample noise
            noise = tf.constant(sample_noise(real_data.shape[0], noise_dim))

            # Compute gradient
            with tf.GradientTape() as tape:
                fake_data = generator(noise)
                loss = g_loss(discriminator(fake_data))
            grads = tape.gradient(loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        else:
            epsilon = 0.01

            # target performance
            real_logit = inv_sigmoid(discriminator(real_data))
            l_min = np.ma.masked_invalid(real_logit).min()
            l_max = np.ma.masked_invalid(real_logit).max()  # max(real_logit)
            print(f"{l_min} {l_max}")
            stop_th = np.random.uniform(l_min, l_max)

            x = tf.Variable(tf.random.normal(shape=(batch_size, data.shape[1])))
            # by updating x we try the classifier to classify the images
            for intro_it in range(1000):
                with tf.GradientTape() as intro_tape:
                    intro_tape.watch(x)
                    hat_y = discriminator(x)
                grads = - intro_tape.gradient(hat_y, x)  # grad acent
                grads = epsilon / 2 * grads + tf.random.normal(shape=grads.shape, mean=0, stddev=epsilon * 0.1)
                g_optimizer.apply_gradients(zip([grads], [x]))

                # test performance
                mean_hat_logit = np.mean(inv_sigmoid(discriminator(x)))
                diff = stop_th - mean_hat_logit
                if diff < 0:
                    break

                epsilon = np.clip(diff, 1e-5, 0.01)

            print(f"Objective {mean_hat_logit}")
            fake_batch = np.vstack((fake_batch, x.numpy()))

    # Visualize
    if it % 1 == 0:
        vis_data(data, lims)

        if TASK < 5:
            vis_g(generator, fixed_noise, lims)
            vis_d(generator, discriminator, fixed_noise, X_grid, Y_grid, grid, g_loss)
        else:
            # UNCOMMENT AND SUPPLY YOUR SAMPLES FOR BONUS TASK 5
            vis_points(fake_batch[-1000:-batch_size], lims)
            vis_points(fake_batch[-batch_size:], lims, c='r')
        plt.show()
        print(f"Task {TASK}; Iteration {it}")




