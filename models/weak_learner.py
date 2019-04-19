#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Implements StumpBase and BestStump as found on page 32 in documentation of MutliBoost.

def stump(j, b):
    """
    Parameters
    ----------
    j : int
        Index of feature to use.

    b : float
        Threshold for the stump.

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
    X: TODO
    Y: TODO
    W: TODO
    returns: alpha, v, phi, gamma; Make sure v is a numpy array

    """

    # Make sure phi = stump(j^*, b_{j^*})
    pass

# This is the helper function for stump_base
def best_stump(s, Y, W, gamma_vec_init):
    """
    s: TODO
    Y: TODO
    W: TODO
    gamma_vec_init: TODO
    returns: v, b, gamma
    """
    pass

