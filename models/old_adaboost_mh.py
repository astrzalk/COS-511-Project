#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# from sklearn.tree import DecisionTreeClassifier

# How to use sklearn tree:
# clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)

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

def old_ada_boost_mh(X_train, y_train, X_test, y_test, T, clf):
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

    Returns
    -------
    err_train, err_test: A tuple of floats representing the training
                         error, and testing error for T rounds of boosting.
    """
    N = X_train.shape[0]
    k = len(np.unique([y_train, y_test]))
    X_train_m, y_train_m = get_multiclass_data(X_train, y_train, k)
    X_test_m, y_test_m = get_multiclass_data(X_test, y_test, k)
    training_error, testing_error = [], []

    D_t = np.ones((N * k, 1)) * (1 / (N * k)) # N * k by 1
    for t in range(T):
        h_t = clf.fit(X_train_m, y_train_m, sample_weight=D_t)
        # r_t = \sum_{i, l} D_t(i, l) Y_i[l] h_t(x_i, l)
        h_t_x_l = np.array([h_t.predict(X_train_m[i]) for i in range(N*k)])
        r_t = np.sum(np.multiply(np.multiply(D_t, y_train_m), h_t_x_l))
        err_train_t = 0.5 * (1 - r_t)
        training_error.append(err_train_t)

        # Get testing error of h_t
        h_t_x_l_test = np.array([h_t.predict(X_test_m[i]) for i in range(N*k)])
        r_t_test = np.sum(np.multiply(np.multiply(D_t, y_test_m), h_t_x_l_test))
        err_test_t = 0.5 * (1 - r_t_test)
        testing_error.append(err_test_t)

        # Update D_t
        alpha_t = 0.5 * np.log((1 + r_t) / (1 - r_t))
        Z_t = np.sqrt(1 - np.square(r_t))
        update = np.exp(-alpha_t * np.multiply(y_train_m, h_t_x_l)) / Z_t
        D_t = np.multiply(D_t, update)
    return (training_error, testing_error)


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






