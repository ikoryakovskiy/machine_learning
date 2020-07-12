#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:24:09 2020

@author: ivan

https://github.com/bayesgroup/deepbayes-2018/blob/master/day2_vae/vae_complete.ipynb

Auto-Encoding Variational Bayes
https://arxiv.org/pdf/1312.6114.pdf
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pylab as plt


class BernoulliVector:
    def __init__(self, logits):
        # self.logits is a $N \times D$-dimensional tensor of logits for each of $D$ pixels of $N$ object in batch.
        self.logits = logits

    def log_prob(self, x):
        """
        Note: This is Bernoulli distribution, not cross-entropy loss, => minus sign is not needed!
        
        Original form is:
            sg = keras.activations.sigmoid(self.logits)
            l_prob = x * tf.math.log(sg) + (1 - x) * tf.math.log(1 - sg)
        """
        # After simplification...
        l_prob = -keras.activations.softplus(-self.logits) - self.logits * (1 - x)
        return tf.math.reduce_sum(l_prob, axis=1)

    def sample(self):
        samples = keras.activations.sigmoid(self.logits) >= tf.random.uniform(shape=self.logits.shape)
        return tf.cast(samples, dtype=np.float32)


def test_BernoulliVector():
    logits = tf.constant([[0.26257313, 1.00010365, 1.32164169, -0.60049884, 0.47478581],
                          [-0.69943423, 0.40572153, 0.91215638, 1.36048238, 0.28434441],
                          [0.11055949, -0.65058279, -1.74598369, 1.2715774, -0.60143489]])
    bv = BernoulliVector(logits)
    # test log_prob()
    x = tf.constant([[0, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 1]], dtype=np.float)
    log_probs = bv.log_prob(x)
    assert(log_probs.shape[0] == 3), 'log_prob() returns wrong shape'

    true_log_probs = np.asarray([-2.3037, -5.0039, -6.2844], dtype=np.float32)
    np.testing.assert_allclose(log_probs, true_log_probs, atol=1e-4,
                               err_msg='log_prob() returns wrong values')
    # test sample()
    assert(logits.shape == bv.sample().shape), 'sample() returns wrong shape'

    mean = np.zeros_like(logits)
    for i in range(1024):
        mean += 1 / 1024 * bv.sample()

    np.testing.assert_allclose(keras.activations.sigmoid(logits), mean, atol=1e-1,
                               err_msg='law of large number seems to be violated by sample()')

    print("All fine!")


class MultivariateNormalDiag:
    def __init__(self, mean=None, stddev=None):
        self.mean = mean
        self.stddev = stddev

    def log_prob(self, x):
        normalization_const = - 0.5 * x.shape[1] * tf.math.log(2 * np.pi)
        normalization_const += - tf.math.reduce_sum(tf.math.log(self.stddev), axis=1)
        sq_term = - 0.5 * tf.math.reduce_sum(((x - self.mean) / self.stddev) ** 2, axis=1)
        l_prob = normalization_const + sq_term
        return l_prob

    def log_prob_easier(self, x):
        """
        By iid assumprion, we can
        """
        normalization_const = -0.5 * tf.math.log(2 * np.pi) - tf.math.log(self.stddev)
        sq_term = - 0.5 * ((x - self.mean) / self.stddev) ** 2
        l_prob = tf.math.reduce_sum(normalization_const + sq_term, axis=1)
        return l_prob

    def sample(self):
        return self.mean + self.stddev * tf.random.normal(shape=self.stddev.shape)


def test_MultivariateNormalDiag():
    mean = tf.constant([[0.0619, 1.9728, 0.2092],
                        [0.3971, -0.1817, 1.1508]])
    stddev = tf.constant([[0.0619, 1.9728, 0.2092],
                          [0.3971, 0.1817, 1.1508]])

    mnd = MultivariateNormalDiag(mean, stddev)
    # test log_prob()
    x = tf.ones_like(mean)
    log_probs = mnd.log_prob(x)
    assert(log_probs.shape[0] == 2), 'log_prob() returns wrong shape'

    true_log_probs = np.asarray([-121.1941, -22.5777], dtype=np.float32)
    np.testing.assert_allclose(log_probs.numpy(), true_log_probs, atol=1e-4,
                               err_msg='log_prob() returns wrong values')
    # test sample()
    assert(mean.shape == mnd.sample().shape), 'sample() returns wrong shape'

    est_mean = tf.zeros_like(mean)
    for i in range(1024):
        est_mean += 1 / 1024 * mnd.sample()

    np.testing.assert_allclose(est_mean, mean, atol=1e-1,
                               err_msg='law of large number seems to be violated by sample()')

    print("All fine!")


test_BernoulliVector()
test_MultivariateNormalDiag()


# ================ DATASET ================ #
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype(np.float32) / 255.0  # float32 is default in TF 2
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

x_test = x_test.reshape((-1, 28 * 28)).astype(np.float32) / 255.0
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# ================ MODEL ================ #
d, nh, D = 32, 100, 28 * 28

encoder = keras.Sequential()
encoder.add(layers.Dense(nh, activation='relu', input_shape=(D,)))
encoder.add(layers.Dense(nh, activation='relu'))
encoder.add(layers.Dense(2 * d))

decoder = keras.Sequential()
decoder.add(layers.Dense(nh, activation='relu', input_shape=(d,)))
decoder.add(layers.Dense(nh, activation='relu'))
decoder.add(layers.Dense(D))

# ================ TRAINING ================ #
def loss(x, encoder, decoder):
    """
    This is maximization problem because we want to maximize probability.
    log p(x) = L(q(z)) + KL(q(z|x) || p(z|x))
        p(x) does not depend on q => to minimize KL-divergence, we need to 
        maximize L(q(z))
    See section 2.2 of Auto-Encoding Variational Bayes (https://arxiv.org/pdf/1312.6114.pdf)
    L(q(z)) = log p(x|z) - KL(q(z|x) || p(z)) = E[ log p(x|z) + log p(z) - log q(z|x) ]
    """
    q_loc_scale = encoder(x)

    # the following enforces prior for the latent variable z ~ N(0, 1)
    qz_x = MultivariateNormalDiag(q_loc_scale[:, :d], tf.math.softplus(q_loc_scale[:, d:]))
    z = qz_x.sample()
    pz = MultivariateNormalDiag(tf.zeros_like(z), tf.ones_like(z))

    px_z_logits = decoder(z)
    px_z = BernoulliVector(px_z_logits)
    mean_loss = tf.math.reduce_mean(px_z.log_prob(x) + pz.log_prob(z) - qz_x.log_prob(z))
    return mean_loss


@tf.function
def step(x, encoder, decoder, optimizer, loss):
    """
    Apply_gradients applies gradients in opposite direction.
    By using "-" in loss function, we are able to achieve maximize loss function instead.
    """
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(trainable_variables)
        lo = loss(x, encoder, decoder)
        minimize_loss = - lo
    grads = tape.gradient(minimize_loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return lo


def train_model(encoder, decoder, batch_size=100, num_epochs=3, learning_rate=1e-3):
    optimizer = keras.optimizers.Adam(learning_rate)
    train_ds = train_dataset.batch(batch_size).shuffle(1000)
    test_ds = test_dataset.batch(batch_size)
    train_losses = []
    test_results = []
    for _ in range(num_epochs):
        for i, batch in enumerate(train_ds.take(batch_size)):
            total = len(y_train)
            lo = step(batch[0], encoder, decoder, optimizer, loss)
            train_losses.append(lo.numpy())
            if (i + 1) % 10 == 0:
                print('\rTrain loss:', train_losses[-1],
                      'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)

        test_elbo = 0
        for i, batch in enumerate(test_ds.take(batch_size)):
            batch_elbo = loss(batch[0], encoder, decoder)
            test_elbo += (batch_elbo - test_elbo) / (i + 1)
        test_results.append(test_elbo)
        print('\nTest loss after an epoch: {}'.format(test_elbo))


train_model(encoder, decoder, num_epochs=16)


# ================ VISUALIZATION ================ #
def plot_reconstructions(encoder, decoder):
    plot_batch = x_test[:25]
    # apply zero noise
    x_hat = tf.math.sigmoid(decoder(encoder(plot_batch)[:, :d])).numpy()
    # We can also do clipping here
    #x_hat = np.clip(decoder(encoder(plot_batch)[:, :d]).numpy(), 0.0, 1.0)
    x_hat = x_hat.reshape((-1, 28, 28))
    batch = plot_batch.reshape((-1, 28, 28))
    
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(14, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(25):
        axes[i % 5, 2 * (i // 5)].imshow(batch[i], cmap='gray')
        axes[i % 5, 2 * (i // 5) + 1].imshow(x_hat[i], cmap='gray')


def plot_interpolations(encoder, decoder):
    batch = encoder(x_test[:10]).numpy()
    z_0 = batch[:5, :d].reshape((5, -1, d))
    z_1 = batch[5:, :d].reshape((5, -1, d))
    alpha = np.linspace(0.0, 1.0, 10).reshape((1, 10, 1))
    interpolations_z = z_0 * alpha + z_1 * (1 - alpha)
    interpolations_z = interpolations_z.reshape((50, d))
    interpolations_x = tf.math.sigmoid(decoder(interpolations_z)).numpy()
    interpolations_x = interpolations_x.reshape(5, 10, 28, 28)
    
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(14, 7),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(50):
        axes[i // 10, i % 10].imshow(interpolations_x[i // 10, i % 10], cmap='gray')
        

def plot_tsne(objects, labels):
    embeddings = TSNE(n_components=2).fit_transform(objects)
    plt.figure(figsize=(8, 8))
    for k in range(10):
        embeddings_for_k = embeddings[labels == k]
        plt.scatter(embeddings_for_k[:, 0], embeddings_for_k[:, 1],
                    label='{}'.format(k))
    plt.legend()


def plot_pca(objects, labels):
    embeddings = PCA(n_components=2).fit(objects).transform(objects)
    plt.figure(figsize=(8, 8))
    for k in range(10):
        embeddings_for_k = embeddings[labels == k]
        plt.scatter(embeddings_for_k[:, 0], embeddings_for_k[:, 1],
                    label='{}'.format(k))
    plt.legend()
    
        
plot_reconstructions(encoder, decoder)
plot_interpolations(encoder, decoder)

z = encoder(x_test)[:1000, :d].numpy()
plot_tsne(z, y_test[:1000])
plot_pca(z, y_test[:1000])