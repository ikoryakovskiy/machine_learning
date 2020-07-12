#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:24:09 2020

@author: ivan

https://github.com/bayesgroup/deepbayes-2018/blob/master/day2_vae/vae_complete.ipynb

Semi-supervised Learning withDeep Generative Models
https://arxiv.org/pdf/1406.5298.pdf
http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm


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

# ================ GUMBEL-SOFTMAX ================ #
class RelaxedOneHotCategorical():
    """
    In machine learning, the Gumbel distribution is sometimes employed to
    generate samples from the categorical distribution.
    Otherwice, we would need to do a regection sampling, but this approach is
    much cheaper computationally.
    https://stats.stackexchange.com/questions/366948/why-do-we-need-the-temperature-in-gumbel-softmax-trick
    https://medium.com/mini-distill/discrete-optimization-beyond-reinforce-5ca171bebf17
    """
    def __init__(self, logits, temperature):
        self.k = tf.constant(logits.shape[1], dtype=tf.float32)
        self.logits = logits
        self.temperature = tf.constant(temperature, dtype=tf.float32)

    def log_prob(self, x):
        log_Z = tf.math.lgamma(self.k) + (self.k - 1) * tf.math.log(self.temperature)
        log_prob_unnormalized = tf.math.log(tf.math.softmax(
            self.logits - self.temperature * tf.math.log(x), axis=1)
            ) - tf.math.log(x)
        return tf.math.reduce_sum(log_prob_unnormalized) + log_Z
    
    def sample(self):
        gumbel = -tf.math.log(-tf.math.log(tf.random.uniform(self.logits.shape)))
        sample = tf.math.softmax((self.logits + gumbel) / self.temperature, axis=1)
        return sample
    

def test_gumbel():
    n_classes = 4
    logits = tf.random.normal((1, n_classes))
    temperatures = [0.1, 0.5, 1., 5., 10.]
    M = 128 # number of samples used to approximate distribution mean
    
    fig, axes = plt.subplots(nrows=2, ncols=len(temperatures), figsize=(14, 6),
                             subplot_kw={'xticks': range(n_classes),
                                         'yticks': [0., 0.5, 1.]})
    axes[0, 0].set_ylabel('Expectation')
    axes[1, 0].set_ylabel('Gumbel Softmax Sample')
    
    for n, t in enumerate(temperatures):
        dist = RelaxedOneHotCategorical(logits, t)
        mean = tf.zeros_like(logits)
        for _ in range(M):
            mean += dist.sample() / M
        sample = dist.sample()
        
        axes[0, n].set_title('T = {}'.format(t))
        axes[0, n].set_ylim((0, 1.1))
        axes[1, n].set_ylim((0, 1.1))
        axes[0, n].bar(np.arange(n_classes), mean.numpy().reshape(n_classes),
                       color=cm.plasma(0.75 * t / max(temperatures)))
        axes[1, n].bar(np.arange(n_classes), sample.numpy().reshape(n_classes),
                       color=cm.plasma(0.75 * t / max(temperatures)))
    plt.show()


test_gumbel()

# ================ DATASET ================ #
def one_hot_encoding(y, n_cat=10):
    """
    one_hot = np.zeros((len(y), n_cat))
    for i in range(len(y)):
        one_hot[i][y[i]] = 1
    """
    return np.eye(n_cat)[y]
    

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype(np.float32) / 255.0  # float32 is default in TF 2
new_train_labels = np.zeros((60000, 10), dtype=np.float32)
observed = np.random.choice(60000, 3000)
new_train_labels[observed] = np.eye(10)[y_train[observed]]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, new_train_labels))

x_test = x_test.reshape((-1, 28 * 28)).astype(np.float32) / 255.0
# y_test = one_hot_encoding(y_test)  # no need to convert to one-hot
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# ================ MODEL ================ #
n_classes, d, nh, D = 10, 32, 500, 28 * 28

z_encoder = keras.Sequential()
z_encoder.add(layers.Dense(nh, activation='relu', input_shape=(n_classes + D,)))
z_encoder.add(layers.Dense(2 * d))

yz_decoder = keras.Sequential()
yz_decoder.add(layers.Dense(nh, activation='relu', input_shape=(n_classes + d,)))
yz_decoder.add(layers.Dense(D))

y_encoder = keras.Sequential()
y_encoder.add(layers.Dense(nh, activation='relu', input_shape=(D,)))
y_encoder.add(layers.Dense(n_classes))

# ================ TRAINING ================ #
def loss(x, y, y_encoder, z_encoder, yz_decoder, T=0.6, alpha=32.):
    ################################################################################
    # NOTE:                                                                        #
    # hyperparameter alpha was tuned for the implementation that computed  mean of #
    # elbo terms and sum of cross-entropy terms over observed datapoints in batch  #
    ################################################################################
    y_is_observed = tf.math.reduce_sum(y, axis=1, keepdims=True)  # creates if-else condition
    # sample y from q(y | x)
    qy_x = RelaxedOneHotCategorical(y_encoder(x), T)
    y_gumbel = qy_x.sample()
    
    # y either comes from gumbel or from the observed values => a concatenation of categorical and softmax
    y_to_decode = y_gumbel * (1 - y_is_observed) + y * y_is_observed
    
    # sample z from q(z | x, y)
    qz_xy_loc_scale = z_encoder(tf.concat([x, y_to_decode], axis=1))
    qz_xy = MultivariateNormalDiag(
        qz_xy_loc_scale[:, :d],
        tf.math.softplus(qz_xy_loc_scale[:, d:]))
    z = qz_xy.sample()    
    # compute the evidence lower bound
    py = RelaxedOneHotCategorical(tf.zeros_like(y_gumbel), T)
    pz = MultivariateNormalDiag(tf.zeros_like(z), tf.ones_like(z))
    px_yz_logits = yz_decoder(tf.concat([y_to_decode, z], axis=1))
    px_yz = BernoulliVector(px_yz_logits)
    
    # #test
    # artificial_logits = tf.tile(tf.eye(10), (10,1)) * (np.e - 1) + 1
    # prior_prob = py.log_prob(artificial_logits)
    
    loss_unsupervised = px_yz.log_prob(x) + pz.log_prob(z) - qz_xy.log_prob(z) \
        + (py.log_prob(y_gumbel) - qy_x.log_prob(y_gumbel)) * (1 - y_is_observed)
    loss_unsupervised = tf.math.reduce_mean(loss_unsupervised)
    # compute the cross_entropy regularizer with weight alpha
    loss_supervised = tf.math.reduce_sum(
        y_is_observed * y * tf.math.log_softmax(y_encoder(x), axis=1)
    )
    return loss_unsupervised + alpha * loss_supervised


@tf.function
def step(x, y, y_encoder, z_encoder, yz_decoder, optimizer, loss):
    """
    Apply_gradients applies gradients in opposite direction.
    By using "-" in loss function, we are able to achieve maximize loss function instead.
    """
    trainable_variables = y_encoder.trainable_variables + z_encoder.trainable_variables \
        + yz_decoder.trainable_variables
    with tf.GradientTape() as tape:
        tape.watch(trainable_variables)
        lo = loss(x, y, y_encoder, z_encoder, yz_decoder)
        minimize_loss = - lo
    grads = tape.gradient(minimize_loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return lo


def train_model(y_encoder, z_encoder, yz_decoder, batch_size=100, num_epochs=3, learning_rate=1e-3):
    optimizer = keras.optimizers.Adam(learning_rate)
    train_ds = train_dataset.batch(batch_size).shuffle(1000)
    test_ds = test_dataset.batch(batch_size)
    train_losses = []
    test_results = []
    for _ in range(num_epochs):
        for i, batch in enumerate(train_ds.take(batch_size)):
            total = len(y_train)
            lo = step(batch[0], batch[1], y_encoder, z_encoder, yz_decoder, optimizer, loss)
            train_losses.append(lo.numpy())
            if (i + 1) % 10 == 0:
                print('\rTrain loss:', train_losses[-1],
                      'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)

        loss_value = 0
        accuracy = 0
        for i, batch in enumerate(test_ds.take(batch_size)):
            total = len(y_test)
            y = tf.zeros((batch[1].shape[0], 10))
            loss_value += loss(batch[0], y, y_encoder, z_encoder, yz_decoder)
            match = tf.argmax(y_encoder(batch[0]), axis=1).numpy() == batch[1].numpy()
            accuracy += np.mean(match.astype(np.float32))
        test_results.append(accuracy)
        print('Test loss: {}\t Test accuracy: {}'.format(loss_value / total, accuracy / total))


train_model(y_encoder, z_encoder, yz_decoder, num_epochs=100)


# ================ VISUALIZATION ================ #
def plot_samples_with_fixed_classes(yz_decoder):
    classes_x10 = tf.repeat(tf.eye(10), 10, axis=0)
    noise = tf.random.normal((100, d))
    decoder_input = tf.concat((classes_x10, noise), axis=1)
    images = tf.math.sigmoid(yz_decoder(decoder_input)).numpy()
    images = np.reshape(images, (100, 28, 28))
    
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(14, 14),
                             subplot_kw={'xticks': [], 'yticks': []})
    for i in range(10):
        axes[0, i].set_title('{}'.format(i))
    for i in range(100):
        axes[int(i / 10), i % 10].imshow(images[i], cmap='gray')
    plt.show()
        

def plot_all_digits_with_fixed_style(z_encoder, yz_decoder):
    indices = np.random.choice(10000, 10)
    x, y = x_test[indices], one_hot_encoding(y_test[indices])
    z = z_encoder(tf.concat((x, y), axis=1))[:, :d]

    # generate digits
    images = []
    for i in range(10):
        digit_encodings = np.tile(np.eye(10)[i], (10, 1))
        new_input = tf.concat((digit_encodings, z), axis=1)
        img = tf.math.sigmoid(yz_decoder(new_input)).numpy()
        img = np.reshape(img, (10, 28, 28))
        images.append(img)
        
    x = np.reshape(x, (10, 28, 28))

    # plot
    fig, axes = plt.subplots(nrows=10, ncols=11, figsize=(14, 14),
                             subplot_kw={'xticks': [], 'yticks': []})
    
    axes[0, 0].set_title('example')
    for i in range(10):
        axes[0, i + 1].set_title('{}'.format(i))
        axes[i, 0].imshow(x[i], cmap='gray')
        for j in range(10):
            axes[i, j + 1].imshow(images[j][i], cmap='gray')
    plt.show()


def plot_tsne(objects, labels):
    embeddings = TSNE(n_components=2).fit_transform(objects)
    plt.figure(figsize=(8, 8))
    for k in range(10):
        embeddings_for_k = embeddings[labels == k]
        plt.scatter(embeddings_for_k[:, 0], embeddings_for_k[:, 1],
                    label='{}'.format(k))
    plt.legend()
    plt.show()
    

plot_samples_with_fixed_classes(yz_decoder)        
plot_all_digits_with_fixed_style(z_encoder, yz_decoder)

# T-SNE for q(z | x, y) mean
labels = y_test[:1000]
encoder_input = tf.concat((x_test[:1000], np.eye(10)[labels]), axis=1)
latent_variables = z_encoder(encoder_input)[:, :d]
latent_variables = latent_variables.numpy()
plot_tsne(latent_variables, labels)

# T-SNE for q(y | x) logits
labels = y_test[:1000]
latent_variables = y_encoder(x_test[:1000]).numpy()
plot_tsne(latent_variables, labels)
