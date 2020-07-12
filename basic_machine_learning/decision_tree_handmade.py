#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:23:14 2020

@author: ivan

Realization of decision tree by me based on:
    https://www.youtube.com/watch?v=LDRbO9a6XPU
    deep_learning/notebooks/decision_tree.ipynb
    https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb

Differences:
    1. I used entropy instead of gini
    2. I used heuristix (median) to select a feature thresholds becasue I did not want to iterate
       over all values of the feature

Multi-class classification is straightforward. I just need to account for all classes in entropy calculation.
"""
import math
from collections import Counter
import numpy as np


class Question:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

    def answer(self, dp):
        return dp[self.feature] >= self.threshold


def split(data, question):
    true_branch_data = []
    false_branch_data = []
    for dp in data:
        if question.answer(dp):
            true_branch_data.append(dp)
        else:
            false_branch_data.append(dp)
    return np.array(true_branch_data), np.array(false_branch_data)


def calculate_entropy(p, q):
    if p is None or q is None:
        return 0
    return - p / (p + q) * math.log2(p / (p + q)) - q / (p + q) * math.log2(q / (p + q))


def calculate_information_gain(hp, true_branch_data, false_branch_data):
    if false_branch_data.size == 0 or true_branch_data.size == 0:
        return 0

    true_branch_labels = Counter(true_branch_data[:, -1])
    ht = calculate_entropy(true_branch_labels.get(0), true_branch_labels.get(1))

    false_branch_labels = Counter(false_branch_data[:, -1])
    hf = calculate_entropy(false_branch_labels.get(0), false_branch_labels.get(1))

    p_true_branch = len(true_branch_data) / (len(true_branch_data) + len(false_branch_data))

    information_gain = hp - p_true_branch * ht - (1 - p_true_branch) * hf

    return information_gain


def find_best_split(parent_data):

    if parent_data.shape[0] < 2:
        return None

    parent_labels = Counter(parent_data[:, -1])
    hp = calculate_entropy(parent_labels.get(0), parent_labels.get(1))

    parent_medians = np.median(parent_data, axis=0)

    best_gain = 0
    best_question = None
    n_features = parent_data.shape[1] - 1
    for feature in range(n_features):
        threshold = parent_medians[feature]
        question = Question(feature, threshold)

        true_branch_data, false_branch_data = split(parent_data, question)

        ig = calculate_information_gain(hp, true_branch_data, false_branch_data)

        if ig > best_gain:
            best_gain = ig
            best_question = question

    return best_question


class Leaf:
    def __init__(self, data):
        """
        labels: key - label, value - probability of that value
        """
        counts = Counter(data[:, -1])
        total_counts = data.shape[0]

        self.labels = {}

        for label in counts:
            self.labels[label] = counts[label] / total_counts


class Node:
    def __init__(self, question=None, true_branch=None, false_branch=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(data):
    best_question = find_best_split(data)

    if best_question is None:
        return Leaf(data)

    true_branch_data, false_branch_data = split(data, best_question)

    node = Node(best_question)

    node.true_branch = build_tree(true_branch_data)
    node.false_branch = build_tree(false_branch_data)

    return node


def predict_label_of_dp(node, dp):
    if isinstance(node, Node):
        if node.question.answer(dp):
            return predict_label_of_dp(node.true_branch, dp)
        else:
            return predict_label_of_dp(node.false_branch, dp)
    else:
        return node.labels


def predict(tree, data):
    labels = []
    for dp in data:
        labels.append(predict_label_of_dp(tree, dp))
    return labels


def predict_map(tree, data):
    labels = []
    for dp in data:
        labels_pr = predict_label_of_dp(tree, dp)
        map_label = max(labels_pr.items(), key=lambda x: x[1])[0]
        labels.append(map_label)
    return labels


# First two digits are X, last one is label y
# Rule: sum(X) > 7 => class 1, else class 0
train_data = [
    [2, 3, 0],
    [3, 2, 0],
    [1, 4, 0],
    [3, 8, 1],
    [8, 0, 1],
    [13, 2, 1],
    [10, 3, 1],
    [0, 5, 0],
    [7, 0, 0],
    [6, 2, 1],
    [2, 3, 1],  # confusing sample / outlier
    [2, 3, 1],  # confusing sample / outlier
]
train_data = np.array(train_data)

tree = build_tree(train_data)


test_data = [
    [0, 0],  # 0
    [3, 4],  # 0
    [10, 0],  # 1
    [5, 5],  # 1
    [2, 3],  # 0 with pr 0.33 and 1 with probability 0.67 (here it learned wrong values indeed)
]
test_data = np.array(test_data)

labels = predict(tree, test_data)
print(labels)

labels_map = predict_map(tree, test_data)
print(labels_map)
