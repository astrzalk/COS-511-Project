#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
# The below code has three main adaboost functions:
# S_adaboost_mh is the original Schapire, Singer adaboost.mh
# interpreted to mean unraveled vector for both Y and W.
##
# K_adaboost_mh is Kegl's interpretation of Schapire's adaboost.mh.
##
# factorized_adaboost_mh is the factorized version of adaboost as proposed
# in return of adaboost paper.
############################################################################

import numpy as np
from sklearn.tree import DecisionTreeClassifier


# NOTE that if there are bugs this function would be the first place to check
# as getting the logic right for generating Y_m is a little tough and I could
# have unraveled the wrong way.
def get_multiclass_data(X, Y, k):
    """
    Input
    ____
    X: N by d, numpy-like array, training data.
    Y: N by 1, numpy-like array, training labels.
    k: Positive int, number of unique labels.

    returns
    -------
    X_m: (N * k) by (d + 1), transformed training data to map each row or example in X
         to k classes and add a new column differentiating each identical row. Meaning
         that two rows can have the same x_1 but differ by the class which is held in the
         last column (this is why we needed d + 1, as opposed to d).
    Y_m: (N * k) by 1, each element is either \pm 1.
         It is the indicator of whether the ith label is equal to a particular class.
    """
    N = X.shape[0]
    # the [:, None] trick is to coerce (n,) -> (n,1).
    class_col = np.arange(k)[:, None]
    repeat_row = lambda x: np.tile(x, (k,1))

    # Each element in Xs is just a repeated row of X, x_i, k times with the additional
    # column of the k classes.
    Xs = [np.hstack((repeat_row(X[i, :]), class_col))
          for i in range(N)]
    X_m = np.vstack(Xs)

    Y = Y[:, None] # Make sure that it is (N, 1) instead of (N,)
    # Ys takes each ith label and repeats it k times,
    # so it will be a k by N matrix where each column is the ith
    # label repeated k times.
    Ys = np.hstack([repeat_row(Y[i, :]) for i in range(N)])
    # Unravel the matrix column wise
    Ys_col = np.ravel(Ys, order='F')
    Y_m = [1 if X_m[i, -1] == Ys_col[i] else -1 for i in range(N*k)]

    return (X_m, np.array(Y_m))

def get_ham_loss(W, h, Y, unravel):
    """
    W: N by k (or (N*k,)) weight matrix, numpy-array.
    h: N by k (or (N*k,)) predictions from strong hypothesis
       predicted on data, numpy-array.
    Y: N by k (or (N*k,)) label matrix, numpy-array.
    unravel: Boolean variable to determine whether to think about
             the multiclass case as (N*k, ) (False) or (N, k) (True).
    :returns: h_loss, float.
    """
    if unravel:
        # unravel columnwise i.e. for each class get all the examples
        W_unraveled = np.ravel(W, order='F')
        Y_unraveled = np.ravel(Y, order='F')
        h_unraveled = np.ravel(h, order='F')
        indicator_bool = (np.sign(h_unraveled) != Y_unraveled)
        # The 1*bool trick makes python bools become int 0-1
        h_loss = np.sum(np.multiply(W_unraveled, 1*indicator_bool))
    else:
        indicator_bool = (np.sign(h) != Y)
        h_loss = np.sum(np.multiply(W, 1*indicator_bool))

    return h_loss

def S_adaboost_mh(X_train, y_train, X_test, y_test, T, clf, W_init):
    """
    Input
    -----
    X: The data assumed to be N (number of examples) by d (features).
       Type: np.array-like.
    y: The labels assumed to be N by 1, takes on one of k unique values.
       Type: np.array-like.
    T: Number of rounds to do boosting.
       Type: Int.
    clf: Base (or weak) classifier.
         Type: sklearn classifier, expected just DecisionTreeClassifier.
    W_init: Boolean; True if using uniform weighting scheme, false if using
            the assymetric scheme given by equation (3) in return of adaboost
            paper.

    Returns
    -------
    train_error, test_error: A tuple of floats representing the training
                         error, and testing error for T rounds of boosting.
    """
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    train_labels = set(y_train)
    test_labels = set(y_test)
    k = len(train_labels.union(test_labels))
    X_train_m, y_train_m = get_multiclass_data(X_train, y_train, k)
    X_test_m, y_test_m = get_multiclass_data(X_test, y_test, k)
    training_error, testing_error = [], []

    if W_init:
        D_t = np.ones((N * k, )) * (1 / (N * k)) # N * k by ,
        w = np.ones((N * k, )) * (1 / (N * k))
        w_test = np.ones((N_test * k, )) * (1 / (N_test * k))
    else:
        # Implemented poorly, TODO make it better
        D_t = np.ones((N, k))
        w = np.ones((N, k))
        for i in range(N):
            for j in range(k):
                if y_train_m[k*i + j] == 1:
                    D_t[i, j] = D_t[i, j] * 0.5 * (1 / N)
                    w[i, j] = w[i, j] * 0.5 * (1 / N)
                else:
                    D_t[i, j] = D_t[i, j] * 0.5 * (1 / (N *(k - 1)))
                    w[i, j] = w[i, j] * 0.5 * (1 / (N *(k - 1)))

    h_ts, h_ts_test, gammas = [], [], []
    for t in range(T):
        print("Round {}".format(t + 1))
        h_t = clf.fit(X_train_m, y_train_m, sample_weight=D_t)
        # r_t = \sum_{i, l} D_t(i, l) Y_i[l] h_t(x_i, l) AKA \gamma_t
        h_t_x_l = h_t.predict(X_train_m) # (N*k, )
        h_t_x_l_test = h_t.predict(X_test_m) # (N*k, )
        gamma_t = np.sum(np.multiply(np.multiply(w, y_train_m), h_t_x_l))
        gammas.append(gamma_t)

        # Update D_t
        alpha_t = 0.5 * np.log((1 + gamma_t) / (1 - gamma_t))
        h_ts.append(alpha_t * h_t_x_l)
        h_ts_test.append(alpha_t * h_t_x_l_test)
        Z_t = np.sqrt(1 - np.square(gamma_t))
        update = np.exp(-alpha_t * np.multiply(y_train_m, h_t_x_l)) / Z_t
        D_t = np.multiply(D_t, update)
    H = sum(h_ts)
    H_test = sum(h_ts_test)
    train_error = get_ham_loss(w, H, y_train_m, False) # unravel = False
    test_error = get_ham_loss(w_test, H_test, y_test_m, False) # unravel = False
    return (train_error, testing_error, gammas)


def one_hot_labels(y, k):
    """
    y: N by 1 numpy-array corresponding to the labels.
    k: float, corresponding to number of unique classes.
    returns: Y, an N by k numpy-array with each row containing
             a 1 for the correct class and -1 otherwise.
    """
    N = y.shape[0]
    Y = np.ones((N, k)) * -1
    for i in range(N):
        Y[i, y[i]] *= -1 # Make the correct class become +1.

    return Y
# TODO: Write function that takes a list of k classifiers, data


def K_adaboost_mh(X_train, y_train, X_test, y_test, T, clf, W_init):
    """
    Input
    -----
    X: The data assumed to be N (number of examples) by d (features).
       Type: np.array-like.
    y: The labels assumed to be N by 1, takes on one of k unique values.
       Type: np.array-like.
    T: Number of rounds to do boosting.
       Type: Int.
    clf: Base (or weak) classifier.
         Type: sklearn classifier, expected just DecisionTreeClassifier.
    W_init: Boolean; True if using uniform weighting scheme, false if using
            the assymetric scheme given by equation (3) in return of adaboost
            paper.

    Returns
    -------
    train_error, test_error: A tuple of floats representing the training
                         error, and testing error for T rounds of boosting.
    """
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    train_labels = set(y_train)
    test_labels = set(y_test)
    k = len(train_labels.union(test_labels))
    Y_train = one_hot_labels(y_train, k)
    Y_test = one_hot_labels(y_test, k)

    # Make the below a function eventually
    if W_init:
        D_t = np.ones((N, k)) * (1 / (N * k)) # N * k by ,
        w = np.ones((N, k)) * (1 / (N * k))
        w_test = np.ones((N_test, k)) * (1 / (N_test * k))
    else:
        D_t = np.ones((N, k)) * 0.5 * (1 / N)
        w = np.ones((N, k)) * 0.5 * (1 / N)
        for i in range(N):
            D_t[i, y_train[i]] *= 1 / (k - 1)

    h_ts, h_ts_test, gammas = [], [], []
    for t in range(T):
        print("Round {}".format(t + 1))
        h_t = [clf.fit(X_train, Y_train[:, i], sample_weight=D_t[:, i])
                for i in range(k)] # list with k classifiers
        # r_t = \sum_{i, l} D_t(i, l) Y_i[l] h_t(x_i, l) AKA \gamma_t
        h_t_x_l = np.array([h_t[i].predict(X_train) for i in range(k)]).T
        h_t_x_l_test = np.array([h_t[i].predict(X_test) for i in range(k)]).T
        gamma_t = np.sum(np.multiply(np.multiply(w, Y_train), h_t_x_l)) # assumes all things are (N, k).
        gammas.append(gamma_t)

        # Update D_t
        alpha_t = 0.5 * np.log((1 + gamma_t) / (1 - gamma_t))
        h_ts.append(alpha_t * h_t_x_l)
        h_ts_test.append(alpha_t * h_t_x_l_test)
        Z_t = np.sqrt(1 - np.square(gamma_t))
        update = np.exp(-alpha_t * np.multiply(Y_train, h_t_x_l)) / Z_t
        D_t = np.multiply(D_t, update)
    H = sum(h_ts)
    H_test = sum(h_ts_test)
    train_error = get_ham_loss(w, H, Y_train, True) # unravel = False
    test_error = get_ham_loss(w_test, H_test, Y_test, True) # unravel = False
    return (train_error, test_error, gammas)

def factorized_adaboost_mh(X_train, y_train, X_test, y_test, T, clf, W_init):
    # TODO implement Kegl's factorized adaboost
    pass

if __name__ == "__main__":
    # Test functionality of get_mutlicass_data function.
    X_train = np.load('../data/pendigits_train_data.npy')
    y_train = np.load('../data/pendigits_train_labels.npy')
    k = len(np.unique(y_train))
    X_m_train, y_m_train = get_multiclass_data(X_train, y_train, k)
    print("The size of X_train is {}. X_m_train should have shape {}."\
.format(X_train.shape, (X_train.shape[0]*k, X_train.shape[1] + 1)))
    print("The size of X_m_train is {}.".format(X_m_train.shape))

    print("The size of y_train is {}. y_m_train should have shape {}."\
.format(y_train.shape, (y_train.shape[0]*k, 1)))
    print("The size of X_m_train is {}.".format(y_m_train.shape))

    # Test basic functionality of S_adaboost_mh function
    # i.e. does it even interpret and run...
    X_test = np.load('../data/pendigits_test_data.npy')
    y_test = np.load('../data/pendigits_test_labels.npy')

    T = 20
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state=1)
    a, b, c = S_adaboost_mh(X_train, y_train, X_test, y_test, T, clf_tree, True)
    a1, b1, c1 = K_adaboost_mh(X_train, y_train, X_test, y_test, T, clf_tree, True)

