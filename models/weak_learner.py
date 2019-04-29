#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Implements StumpBase and BestStump as found on page 32 in documentation of MutliBoost docs
# found at http://www.multiboost.org/download documentation.pdf.

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

def get_alpha_and_energy(N, W, Y, phi, X, v):
    """Calcualtes alpha and energy a la kegl.

    Parameters
    ----------
    N : Number of training examples
    W : N x k, weight matrix
    Y : N x k, label matrix
    phi : decision stump
    X : N x d, training data matrix
    v : k x 1, vote vector

    Returns
    -------
    (alpha, Z) : tuple of floats
                 Calculates alpha and energy the way
                 kegl does it in the multiboost code.
    """
    phi_pred = np.array([phi(X[i, :]) * v for i in range(N)])
    pre_mask = np.multiply(phi_pred, Y)
    mask_pls = (pre_mask + 1) / 2
    mask_neg = ((pre_mask - 1) / 2) * -1
    eps_pls = np.sum(np.multiply(mask_pls, W))
    eps_min = np.sum(np.multiply(mask_neg, W))
    alpha = 0.5 * np.log((eps_pls) / (eps_min))
    Z = 2 * np.sqrt(eps_min * eps_pls) + (1 - eps_min - eps_pls)
    return (alpha, Z)

# This is the main function that implements the base learner
def stump_base(X,Y,W):
    """
    X: N by d, numpy-like array, training data
    Y: N by k, numpy-like array, label matrix
    W: N by k, numpy-like array, weight matrix
    returns: alpha, v, phi, gamma; Make sure v is a numpy array

    """

    N = X.shape[0]
    d = X.shape[1]
    k = Y.shape[1]

    # Edge of constant classifier
    gamma_vec_init = np.sum(np.multiply(Y,W), axis = 0)

    b = -np.inf
    j_best = 0
    # Iterate across features and keep the stump that minimizes the energy Z
    Z_best = np.inf
    for j in range(d):
        s = np.sort(X[:,j]) # Get sorted column j to find best threshold to split on
        # Reorder rows of Y and W by sorted order of jth column of X
        inds = np.argsort(X[:, j])
        Y_s = Y[inds, :]
        W_s = W[inds, :]

        (v, b, gamma) = best_stump(s, Y_s, W_s, gamma_vec_init)
        phi = stump(j, b)
        alpha, Z = get_alpha_and_energy(N, W, Y, phi, X, v)

        if Z < Z_best:
            v_best = v
            phi_best = phi
            b_best = b
            alpha_best = alpha
            gamma_best = gamma
            Z_best = Z
            j_best = j
    return (alpha_best, Z_best, v_best, phi_best, gamma_best, b_best, j_best)

# This is the helper function for stump_base
def best_stump(s, Y, W, gamma_vec_init):
    """
    s: N by 1, numpy-like array, sorted column of X
    Y: N by k, numpy-like array, label matrix
    W: N by k, numpy-like array, weight matrix
    gamma_vec_init: N by 1, numpy-like array, edge of constant classifer
    returns: v, b, gamma
    """
    gamma_vec_best = np.copy(gamma_vec_init)
    gamma_vec = np.copy(gamma_vec_init)
    b_best = -1 * np.inf

    n = Y.shape[0]
    k = Y.shape[1]
    W_Y = np.multiply(W, Y)
    for i in range(n-1):
        gamma_vec = gamma_vec - 2.0 * W_Y[i, :]

        if s[i] != s[i+1]:
            if np.sum(np.abs(gamma_vec)) > np.sum(np.abs(gamma_vec_best)):
                gamma_vec_best = np.copy(gamma_vec)
                b_best = 0.5 * (s[i] + s[i+1])

    v_best = np.sign(np.copy(gamma_vec_best))
    return (v_best, b_best, np.sum(np.abs(gamma_vec_best)))
