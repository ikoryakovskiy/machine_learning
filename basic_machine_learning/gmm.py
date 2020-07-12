#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:50:30 2020

@author: ivan

Based on:
    http://bjlkeng.github.io/posts/the-expectation-maximization-algorithm/
    https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def scatter_plot(class1, class2, shape):
    ax = plt.subplot(shape)
    ax.scatter(class1[:, 0], class1[:, 1])
    ax.scatter(class2[:, 0], class2[:, 1])
    return ax


def generate_data(n):
    mean = np.array([-3, -3])
    cov = np.array([[2, 0.5], [0.5, 1]])
    class1 = np.random.multivariate_normal(mean, cov, n)

    mean = np.array([3, 0])
    cov = np.array([[1, -0.5], [-0.5, 2]])
    class2 = np.random.multivariate_normal(mean, cov, n)

    scatter_plot(class1, class2, 121)

    data = np.vstack((class1, class2))
    np.random.shuffle(data)
    return data


class GMM:
    """ Gaussian Mixture Model

    Parameters
    -----------
        k: int , number of gaussian distributions

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200

    Attributes
    -----------
       centroids: array, k, number_features

       cluster_labels: label for each data point

    """

    def __init__(self, C, n_runs):
        self.C = C  # number of Guassians/clusters
        self.n_runs = n_runs

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, X, prediction):
        """Calculate means and covariance of different
            clusters from k-means prediction

        Parameters:
        ------------
        prediction: cluster labels from k-means

        X: N*d numpy array data points

        Returns:
        -------------
        intial_means: for E-step of EM algorithm

        intial_cov: for E-step of EM algorithm

        """
        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)  # returns indices
            self.initial_pi[counter] = len(ids[0]) / X.shape[0]
            self.initial_means[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[counter, :]
            Nk = X[ids].shape[0]  # number of data points in current gaussian
            self.initial_cov[counter, :, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter += 1
        assert np.sum(self.initial_pi) == 1

        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, X):
        """Implement k-means to find starting
            parameter values.
            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster
        Parameters:
        ------------
        X: numpy array of data points

        Returns:
        ----------
        tuple containing initial means and covariance

        _initial_means: numpy array: (C*d)

        _initial_cov: numpy array: (C,d*d)


        """
        # initialize two random means
        mean1 = X[0]
        mean2 = X[1]

        while True:
            prev_mean1 = mean1
            # calculate responsibilities
            resp1 = np.linalg.norm(data - mean1, axis=1)
            resp2 = np.linalg.norm(data - mean2, axis=1)

            class1 = data[resp1 <= resp2]
            class2 = data[resp1 > resp2]

            mean1 = np.mean(class1, axis=0)
            mean2 = np.mean(class2, axis=0)

            if np.linalg.norm(mean1 - prev_mean1) < 0.001:
                break

        prediction = []
        for x in X:
            if np.linalg.norm(mean1 - x) < np.linalg.norm(mean2 - x):
                prediction.append(0)
            else:
                prediction.append(1)

        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)

        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self, X, pi, mu, sigma):
        """Performs E-step on GMM model
        Parameters:
        ------------
        X: (N x d), data points, m: no of features
        pi: (C), weights of mixture components
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices
        Returns:
        ----------
        gamma: (N x C), probabilities of clusters for objects
        """
        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule
            self.gamma[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self, X, gamma):
        """Performs M-step of the GMM
        We need to update our priors, our means
        and our covariance matrix.
        Parameters:
        -----------
        X: (N x d), data
        gamma: (N x C), posterior distribution of lower bound
        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        C = self.gamma.shape[1]  # number of clusters

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)

        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for c in range(C):
            x = X - self.mu[c, :]  # (N x d)

            gamma_diag = np.diag(self.gamma[:, c])
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][c]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self, X, pi, mu, sigma):
        """Computes lower bound loss function

        Parameters:
        -----------
        X: (N x d), data

        Returns:
        ---------
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.gamma[:, c] * (
                np.log(self.pi[c] + 0.00001) + dist.logpdf(X) - np.log(self.gamma[:, c] + 0.000001)
            )
        self.loss = np.sum(self.loss)
        return self.loss

    def fit(self, X):
        """Compute the E-step and M-step and
            Calculates the lowerbound

        Parameters:
        -----------
        X: (N x d), data

        Returns:
        ----------
        instance of GMM

        """
        self.mu, self.sigma, self.pi = self._initialise_parameters(X)

        try:
            for run in range(self.n_runs):
                self.gamma = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)

                if run % 10 == 0:
                    print("Iteration: %d Loss: %0.6f" % (run, loss))

        except Exception as e:
            print(e)

        return self

    def predict(self, X):
        """Returns predicted labels using Bayes Rule to
        Calculate the posterior distribution

        Parameters:
        -------------
        X: ?*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on
        highest responsibility gamma.

        """
        labels = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            labels[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])
        labels = labels.argmax(1)
        return labels

    def predict_proba(self, X):
        """Returns predicted labels

        Parameters:
        -------------
        X: N*d numpy array

        Returns:
        ----------
        labels: predicted cluster based on
        highest responsibility gamma.

        """
        post_proba = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule, try and vectorise
            post_proba[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        return post_proba


data = generate_data(500)

gmm = GMM(2, 500)
gmm.fit(data)
pred = gmm.predict(data)

class1 = data[pred == 0]
class2 = data[pred == 1]

ax = scatter_plot(class1, class2, 122)