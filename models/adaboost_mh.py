#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class AdaBoostMH:

    """
    A class that implements three versions of multiclass adaboost.
    run_schapire: Implements the version of adaboost_mh as originally
                   proposed in "Improved Boosting Algorithms
                   Using Confidence-rated Predictions" by Schapire and Singer 1999.
                   Specifically, it reduces the mutliclass
                   case to binary case by taking each example (x_i, Y_i) and
                   replace it by k examples ((x_i, l), Y_i[l]).
    run_kegl: Implements the version of adaboost_mh as interpreted in
                   "The return of ADABOOST.MH: multi-class Hamming trees" by
                   K\'egl 2013. In particular, trains each label independently.

    run_factorized: Implements the version of adaboost_mh proposed in
                            "The return of ADABOOST.MH: multi-class Hamming trees"
                            by K\'egl 2013, the version such that the weak hypothesis can
                            be factorized by a coefficient alpha, a vote vector v,
                            and weak hypothesis.
    """

    def __init__(self, X_train, y_train, X_test, y_test, bias=0.5):
        self.X_tr, self.y_tr = X_train, y_train
        self.X_te, self.y_te = X_test, y_test
        self.n_tr, self.n_te = X_train.shape[0], X_test.shape[0]
        self.k = y_train.shape[1] # Number of unique classes
        self.b = bias
        self.smoothing_val = 1 / (self.n_tr * 0.01)
        self.w_init_tr = self._get_init_distr('unif', False, True, bias)
        self.w_init_te = self._get_init_distr('unif', False, False, bias)


    def _get_init_distr(self, init_scheme, raveled, use_train, bias):
        """ Computes initial distribution over examples and labels.

        Parameters
        ----------
        init_scheme: Str
                     'unif' if using uniform initialization scheme,
                     'bal' if using balanced initialization scheme,
                     'asym' if using unorthodox init scheme,
                     'rand' if using random init scheme
        raveled: Boolean
                 True if we need to return (n*k, ) output,
                 False if we need to return (n, k) output.
        use_train: Boolean
                   True if using number of training examples,
                   False if using number of testing examples.
        bias: float
                   Float value in [0,1] that assigns the amount of weight to put
                   on the true label.
        Output
        ------
        W: numpy-array, either (n*k,) or (n,k)
           The initial distribution over training examples and labels.
        """
        if use_train:
            n = self.n_tr
        else:
            n = self.n_te
        k = self.k

        if use_train:
            Y = self.y_tr
        else:
            Y = self.y_te

        if init_scheme == 'unif':
            W = np.ones((n, k)) * (1 / (n * k))
        elif init_scheme == 'bal':
            W = np.ones((n, k)) * 0.5 * (1 / (n * (k - 1)))
            if use_train:
                for i in range(n):
                    for l in range(k):
                        if Y[i, l] == 1.0:
                            W[i,l] *= (k - 1)
        elif init_scheme == 'asym':
            W = np.ones((n, k)) * (1-bias) * (1 / (n * (k - 1)))
            if use_train:
                for i in range(n):
                    for l in range(k):
                        if Y[i, l] == 1.0:
                            W[i, l] *= (((k - 1) / (1-bias)) * bias)
        elif init_scheme == 'rand':
            W = np.random.rand(n, k)
            W = W / np.sum(W)

        if raveled:
            W = np.ravel(W, order='F')
            W = W.reshape((W.shape[0],)) # guarantee that it is (n*k, )
        return W

    def _get_ham_loss(self, W, h, Y, unravel):
        """
        W: N by k (or (N*k,)) weight matrix, numpy-array.
        h: N by k (or (N*k,)) predictions from strong hypothesis
           predicted on data, numpy-array.
        Y: N by k (or (N*k,)) label matrix, numpy-array.
        unravel: Boolean variable to determine whether to think about
                 the multiclass case as (N*k, ) (False) or (N, k) (True).
        returns: h_loss, float.
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


    def run_one_against_all(self, T, weak_learner, W_init, verbose=0):
        # Map instance variables to local variables to be more explicit.
        X_train, X_test = self.X_tr, self.X_te
        Y_train, Y_test = self.y_tr, self.y_te
        n_tr, n_te, k = self.n_tr, self.n_te, self.k
        bias = self.b

        # Compute initial distributions
        raveled = False # False in Factorized Interpretation
        init_d_t_train = True # Use training data
        D_t = self._get_init_distr(W_init, raveled, init_d_t_train, bias)

        h_ts_tr, h_ts_te, gammas, D_ts, = [], [], [], [D_t]
        train_errs, test_errs = [], []
        for t in range(T):
            if verbose in (1,2):
                print("Round {}".format(t + 1), flush=True)

            # Fit weak learner to data
            h_t = []
            alphas_t = []
            gamma_t = 0.0
            for l in range(k):
                alpha, _, phi, gamma, b, j = weak_learner(X_train, np.atleast_2d(Y_train[:, l]).T, np.atleast_2d(D_t[:, l]).T)
                #print("{}   gamma = {}  b = {},  j = {}".format(l,gamma,b,j))
                #print("Sum of weights in column = {}".format(np.sum(D_t[:,l])))
                alphas_t.append(alpha)
                h_t.append(phi)
                gamma_t = gamma_t + gamma
            gammas.append(gamma_t)

            # Use weak learner to make predictions on train and test
            h_t_tr = np.array([[alphas_t[l] * h_t[l](X_train[i, :]) for l in range(k)] for i in range(n_tr)])
            h_t_te = np.array([[alphas_t[l] * h_t[l](X_test[i, :]) for l in range(k)] for i in range(n_te)])
            h_ts_tr.append(h_t_tr)
            h_ts_te.append(h_t_te)

            # Update D_t
            if verbose == 2:
                print("alpha is {}\nEdge is {}\n".format(alphas_t, gamma_t))
            update = np.exp(-1 * np.multiply(Y_train, h_t_tr))
            D_t = np.multiply(D_t, update)
            D_t /= np.sum(D_t)
            D_ts.append(D_t)

            # Get error
            H = sum(h_ts_tr)
            H_test = sum(h_ts_te)
            # Calculate the error of H
            # We could make the below cleaner by implementing a _get_error
            # method. It just needs W_init to become an instance variable.
            train_error = self._get_ham_loss(self.w_init_tr, H, Y_train, unravel=True)
            test_error = self._get_ham_loss(self.w_init_te, H_test, Y_test, unravel=True)
            train_errs.append(train_error)
            test_errs.append(test_error)
        return (train_errs, test_errs, gammas, D_ts)








    def run_factorized(self, T, weak_learner, W_init, verbose=0):
        # Map instance variables to local variables to be more explicit.
        X_train, X_test = self.X_tr, self.X_te
        Y_train, Y_test = self.y_tr, self.y_te
        n_tr, n_te, k = self.n_tr, self.n_te, self.k
        bias = self.b

        # Compute initial distributions
        raveled = False # False in Factorized Interpretation
        init_d_t_train = True # Use training data
        D_t = self._get_init_distr(W_init, raveled, init_d_t_train, bias)

        h_ts_tr, h_ts_te, gammas, D_ts, v_ts = [], [], [], [D_t], []
        train_errs, test_errs = [], []
        for t in range(T):
            if verbose in (1,2):
                print("Round {}".format(t + 1), flush=True)
            # Fit weak learner to data
            alpha_t, energy_t, v, phi, gamma_t, b, j = weak_learner(X_train, Y_train, D_t)
            v_ts.append(v)
            assert (isinstance(v, np.ndarray)), "Make v a numpy array."
            gammas.append(gamma_t)

            # Use weak learner to make predictions on train and test
            h_t_tr = np.array([alpha_t * phi(X_train[i, :]) * v for i in range(n_tr)])
            h_t_te = np.array([alpha_t * phi(X_test[i, :]) * v for i in range(n_te)])
            h_ts_tr.append(h_t_tr)
            h_ts_te.append(h_t_te)

            # Update D_t
            if verbose == 2:
                 print("alpha is {}\nEdge is {}\nEnergy is {}\nv is {}\nb is {}\n col is {}".format(alpha_t, gamma_t, energy_t, v, b, j))
            update = np.exp(-1 * np.multiply(Y_train, h_t_tr))
            D_t = np.multiply(D_t, update)
            D_t /= np.sum(D_t)
            D_ts.append(D_t)

            # Get error
            H = sum(h_ts_tr)
            H_test = sum(h_ts_te)
            # Calculate the error of H
            # We could make the below cleaner by implementing a _get_error
            # method. It just needs W_init to become an instance variable.
            train_error = self._get_ham_loss(self.w_init_tr, H, Y_train, unravel=True)
            test_error = self._get_ham_loss(self.w_init_te, H_test, Y_test, unravel=True)
            train_errs.append(train_error)
            test_errs.append(test_error)
        return (train_errs, test_errs, gammas, v_ts, D_ts)
