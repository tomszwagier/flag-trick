import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits


def digits_rsr(n_in, n_out):
    digits = load_digits(n_class=8)
    data_X, data_y = digits.data, digits.target
    X_0 = data_X[digits.target == 0].T
    X_other = data_X[digits.target != 0].T
    X_in = X_0[:, np.random.choice(X_0.shape[1], n_in, replace=False)]
    X_out = X_other[:, np.random.choice(X_other.shape[1], n_out, replace=False)]
    X, y = np.concatenate([X_in, X_out], axis=1), np.array([0] * n_in + [1] * n_out)
    center = np.mean(X_in, axis=1)[:, np.newaxis]
    X = X - center
    return X, y, center


def synthetic_rsr(n_in, n_out):
    center = np.zeros((3,))
    X_in = np.random.multivariate_normal(center, np.diag([5, 1, .1]), size=n_in).T
    X_out = np.random.multivariate_normal(center, np.diag([.1, .1, 5]), size=n_out).T
    X, y = np.concatenate([X_in, X_out], axis=1), np.array([0] * n_in + [1] * n_out)
    return X, y
    

def synthetic_tr_gmm(n):
    X_1 = np.random.multivariate_normal(np.array([-1, -1, 0]), .1 * np.diag([1, 1, 10]), size=n//5).T
    X_2 = np.random.multivariate_normal(np.array([+1, -1, 0]), .1 * np.diag([1, 1, 10]), size=n//5).T
    X_3 = np.random.multivariate_normal(np.array([-2, +1, 0]), .1 * np.diag([1, 1, 10]), size=n//5).T
    X_4 = np.random.multivariate_normal(np.array([+0, +1, 0]), .1 * np.diag([1, 1, 10]), size=n//5).T
    X_5 = np.random.multivariate_normal(np.array([+2, +1, 0]), .1 * np.diag([1, 1, 10]), size=n//5).T
    X, y = np.concatenate([X_1, X_2, X_3, X_4, X_5], axis=1), np.array(([0] * (n//5)) + ([1] * (n//5)) + ([2] * (n//5)) + ([3] * (n//5)) + ([4] * (n//5)))
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    return X, y


def synthetic_tr_rings(n):
    ring = np.array([np.cos(np.linspace(0, 2 * np.pi, n//5)), np.sin(np.linspace(0, 2 * np.pi, n//5)), np.zeros(n//5)])
    X_1 = np.array([-1, -.5, 0])[:, np.newaxis] + ring/2 + np.array([np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.5, size=n//5)])
    X_2 = np.array([+1, -.5, 0])[:, np.newaxis] + ring/2 + np.array([np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.5, size=n//5)])
    X_3 = np.array([-2, +.5, 0])[:, np.newaxis] + ring/2 + np.array([np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.5, size=n//5)])
    X_4 = np.array([+0, +.5, 0])[:, np.newaxis] + ring/2 + np.array([np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.5, size=n//5)])
    X_5 = np.array([+2, +.5, 0])[:, np.newaxis] + ring/2 + np.array([np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.1, size=n//5), np.random.normal(scale=.5, size=n//5)])
    X, y = np.concatenate([X_1, X_2, X_3, X_4, X_5], axis=1), np.array(([0] * (n//5)) + ([1] * (n//5)) + ([2] * (n//5)) + ([3] * (n//5)) + ([4] * (n//5)))
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    return X, y


def synthetic_sc_moon(n):
    outer_circ = np.array([np.cos(np.linspace(0, np.pi, n//2)), np.sin(np.linspace(0, np.pi, n//2)), np.zeros(n//2)]) + np.random.normal(scale=.1, size=(3, n//2))
    inner_circ = np.array([1 - np.cos(np.linspace(0, np.pi, n//2)), 1 - np.sin(np.linspace(0, np.pi, n//2)) - 2, np.zeros(n//2)]) + np.random.normal(scale=.1, size=(3, n//2))
    X, y = np.concatenate([outer_circ, inner_circ], axis=1), np.array(([0] * (n//2)) + ([1] * (n//2)))
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    return X, y
