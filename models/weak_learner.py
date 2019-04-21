#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Implements StumpBase and BestStump as found on page 32 in documentation of MutliBoost.

def stump(j, b):
    """
    Parameters
    ----------
    j : int. Index of feature to use.

    b : float. Threshold for the stump.

    Returns
    -------
    a function
        Takes in a list (or generically a 1-d numpy-array)
        and outputs 1 or -1.
    """
    def phi(x):
        """
        Parameters
        ----------
        x : 1-d numpy-array or list.

        Returns
        -------
        int, \pm 1
        """
        warning_message = "Dimension of x is smaller than the index j being split on."
        assert (len(x) - 1 >= j), warning_message
        return 1 if x[j] >= b else -1
    return phi


# This is the main function that implements the base learner
def stump_base(X,Y,W):
    """
    X: N by d, numpy-like array, training data
    Y: N by k, numpy-like array, label matrix
    W: N by k, numpy-like array, weight matrix
    returns: alpha, v, phi, gamma; Make sure v is a numpy array

    """

    # Make sure phi = stump(j^*, b_{j^*})
    d = X.shape[1]
    k = Y.shape[1]

    # Edge of constant classifier
    gamma_vec_init = np.sum(np.multiply(Y,W), axis = 0)

    # Iterate across features and keep the stump that minimizes the energy Z
    best_Z = np.inf
    for j in range(d):
        s = np.sort(X[:,j]) # Get sorted column j to find best threshold to split on

        (v, b, gamma) = best_stump(s, Y, W, gamma_vec_init)
        alpha = 0.5 * np.log((1 + gamma) / (1 - gamma))
        phi = stump(j, b)
        Z = np.sqrt(1 - np.square(gamma)) # TODO: Not sure about this step

        if Z < best_Z:
            v_best = v
            phi_best = phi
            alpha_best = alpha
            gamma_best = gamma
            best_Z = Z

    return (alpha_best, v_best, phi_best, gamma_best)

# This is the helper function for stump_base
def best_stump(s, Y, W, gamma_vec_init):
    """
    s: N by 1, numpy-like array, sorted column of X
    Y: N by k, numpy-like array, label matrix
    W: N by k, numpy-like array, weight matrix
    gamma_vec_init: N by 1, numpy-like array, edge of constant classifer
    returns: v, b, gamma
    """
    l1_norm = lambda x: np.linalg.norm(x, ord=1)
    gamma_vec_best = np.copy(gamma_vec_init)
    gamma_vec = np.copy(gamma_vec_init)
    b_best = -1 * np.inf

    n = Y.shape[0]
    k = Y.shape[1]
    W_Y = np.multiply(W, Y)
    for i in range(n-1):
        gamma_vec = gamma_vec - 2.0 * W_Y[i, :]

        if s[i] != s[i+1]:
            if l1_norm(gamma_vec) > l1_norm(gamma_vec_best):
                gamma_vec_best = np.copy(gamma_vec)
                b_best = 0.5 * (s[i] + s[i+1])

    v_best = np.sign(np.copy(gamma_vec_best))
    return (v_best, b_best, l1_norm(gamma_vec_best))
