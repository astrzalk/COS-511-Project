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

    def __init__(self, X_train, y_train, X_test, y_test, bias=[0.5,0.5]):
        self.X_tr, self.y_tr = X_train, y_train
        self.X_te, self.y_te = X_test, y_test
        self.n_tr, self.n_te = X_train.shape[0], X_test.shape[0]
        self.k = len(set(y_train).union(set(y_test))) # Number of unique classes
        self.b = bias
        self.smoothing_val = 1 / (self.n_tr * 0.01)

    def _get_multiclass_data(self, X, Y):
        """
        Input
        ____
        X: N by d, numpy-like array, training data.
        Y: N by 1, numpy-like array, training labels.

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
        class_col = np.arange(self.k)[:, None]
        repeat_row = lambda x: np.tile(x, (self.k,1))

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
        Y_m = [1 if X_m[i, -1] == Ys_col[i] else -1 for i in range(N*self.k)]
        return (X_m, np.array(Y_m))

    def _one_hot_labels(self, y):
        """
        y: N by 1 numpy-array corresponding to the labels.
        k: float, corresponding to number of unique classes.
        returns: Y, an N by k numpy-array with each row containing
                 a 1 for the correct class and -1 otherwise.
        """
        N = y.shape[0]
        Y = np.ones((N, self.k)) * -1
        for i in range(N):
            Y[i, y[i]] *= -1 # Make the correct class become +1.
        return Y

    def _get_init_distr(self, init_scheme, raveled, use_train, bias):
        """ Computes initial distribution over examples and labels.

        Parameters
        ----------
        init_scheme: Str
                     'unif' if using uniform initialization scheme,
                     'bal' if using balanced initialization scheme,
                     'asym' if using unorthodox init scheme, assumes k = 2.
        raveled: Boolean
                 True if we need to return (n*k, ) output,
                 False if we need to return (n, k) output.
        use_train: Boolean
                   True if using number of training examples,
                   False if using number of testing examples.
        bias: list
              A 2-element list that contains which initial weights
              get placed on each class, for example bias = [0.5, 0.5]
              the default assumes uniform weighting.
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

        if init_scheme == 'unif':
            W = np.ones((n, k)) * (1 / (n * k))
        elif init_scheme == 'bal':
            W = np.ones((n, k)) * 0.5 * (1 / (n * (k - 1)))
            if use_train:
                for i in range(n):
                    W[i, self.y_tr[i]] *= (k - 1)
            else:
                for i in range(n):
                    W[i, self.y_te[i]] *= (k - 1)
        elif init_scheme == 'asym':
            W = np.ones((n, 2)) * (1 / n) * bias

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


    def _get_alpha_and_energy(self, gamma):
        """Calcualtes alpha and energy a la kegl.

        Parameters
        ----------
        gamma : float
                Calculated in weak_learner.

        Returns
        -------
        (alpha, Z) : tuple of floats
                     Calculates alpha and energy the way
                     kegl does it in the multiboost code.
        """
        # In kegl's code eps_min and eps_pls are defined as
        # eps_min: The error rate of the weak learner, and
        # eps_pls: The correct rate of the weak learner.
        # These two variables seem to act like 1 - gamma, and 1 + gamma
        # respectively, so that is how I am treating them.
        # The corresponding code in Multiboost is located on lines 150-200
        # in MutliBoost/src/WeakLearners/BaseLearner.cpp.
        delta = self.smoothing_val
        eps_pls = 1 + gamma
        eps_min = 1 - gamma
        alpha = 0.5 * np.log((eps_pls + delta) / (eps_min + delta))
        Z = 2 * np.sqrt(eps_min * eps_pls) + (1 - eps_min - eps_pls)
        return (alpha, Z)

    # DON'T LOOK HERE
    def run_schapire(self, T, clf, W_init, verbose=False):
        """
        Input
        -----
        T: Int
           Number of rounds to do boosting.
        clf: sklearn classifier, expected just DecisionTreeClassifier
             Base (or weak) classifier.
        W_init: Boolean
                True if using uniform weighting scheme, false if using
                the assymetric scheme given by equation (3) in return of adaboost
                paper.
        verbose: Boolean
                 True if you want to print rounds in training,
                 False otherwise.
        Returns
        -------
        train_error: float
                     The training error for T rounds of boosting.
        test_error: float
                     The testing error for T rounds of boosting.
        gammas: list
                All T gammas calculated during boosting.
        """
        # Transform datasets so that data goes from (N, d) -> (N*k, d + 1)
        # and labels go from (N, 1) -> (N*k, 1).
        X_train_m, y_train_m = self._get_multiclass_data(self.X_tr, self.y_tr)
        X_test_m, y_test_m = self._get_multiclass_data(self.X_te, self.y_te)
        # Compute Initial Distributions
        raveled = True # True in original interpretation.
        D_t = self._get_init_distr(W_init, raveled, use_train=True)

        h_ts, h_ts_te, gammas, D_ts = [], [], [], [D_t]
        for t in range(T):
            if verbose:
                print("Round {}".format(t + 1))

            h_t = clf.fit(X_train_m, y_train_m, sample_weight=D_t)

            # Calculate gamma_t which is defined in original paper as
            # \sum_{i, l} D_t(i, l) Y_i[l] h_t(x_i, l)
            h_t_x_l = h_t.predict(X_train_m) # (N*k, )
            h_t_x_l_te = h_t.predict(X_test_m) # (N*k, )
            gamma_t = np.sum(np.multiply(np.multiply(D_t, y_train_m), h_t_x_l))
            gammas.append(gamma_t)

            # Update D_t
            alpha_t = 0.5 * np.log((1 + gamma_t) / (1 - gamma_t))
            h_ts.append(alpha_t * h_t_x_l)
            h_ts_te.append(alpha_t * h_t_x_l_te)
            Z_t = np.sqrt(1 - np.square(gamma_t))
            update = np.exp(-alpha_t * np.multiply(y_train_m, h_t_x_l)) / Z_t
            D_t = np.multiply(D_t, update)
            D_ts.append(D_t)
        H = sum(h_ts)
        H_test = sum(h_ts_te)

        # Calculate the error of H
        w_init_tr = self._get_init_distr(W_init, raveled, use_train=True)
        w_init_te = self._get_init_distr(W_init, raveled, use_train=False)
        train_error = self._get_ham_loss(w_init_tr, H, y_train_m, unravel=False)
        test_error = self._get_ham_loss(w_init_te, H_test, y_test_m, unravel=False)
        return (train_error, test_error, gammas, D_ts)

    # DON'T LOOK HERE
    def run_kegl(self, T, clf, W_init, verbose=False):
        """
        Input
        -----
        T: Int
           Number of rounds to do boosting.
        clf: sklearn classifier, expected just DecisionTreeClassifier.
             Base (or weak) classifier.
        W_init: Boolean
                True if using uniform weighting scheme, false if using
                the assymetric scheme given by equation (3) in return of adaboost
                paper.
        verbose: Boolean
                 True if you want to print each round as it starts,
                 False otherwise.
        Returns
        -------
        train_error: float
                     The training error for T rounds of boosting.
        test_error: float
                     The testing error for T rounds of boosting.
        gammas: list
                All T gammas calculated during boosting.
        """
        # Map instance variables to local variables to be more explicit.
        X_train, X_test = self.X_tr, self.X_te
        Y_train, Y_test = self._one_hot_labels(self.y_tr), self._one_hot_labels(self.y_te)
        k = self.k

        # Compute initial distributions
        raveled = False # False in Kegl's interpretation.
        D_t = self._get_init_distr(W_init, raveled, use_train=True)

        h_ts, h_ts_test, gammas, D_ts = [], [], [], [D_t]
        for t in range(T):
            if verbose:
                print("Round {}".format(t + 1))

            # Fit weak learner to data
            h_t = clf(X_train, Y_train, D_t)

            # Check that the below two are arrays of size N_tr(te) by k
            # might need to add a transpose to the end
            n = X_train.shape[0]
            h_t_x_l = np.zeros((n,k))
            for i in range(n):
                for l in range(k):
                    h_t_x_l[i,l] = h_t[l](X_train[i,:])

            n = X_test.shape[0]
            h_t_x_l_test = np.zeros((n,k))
            for i in range(n):
                for l in range(k):
                    h_t_x_l_test[i,l] = h_t[l](X_test[i,:])

            gamma_t = np.sum(np.multiply(np.multiply(D_t, Y_train), h_t_x_l))
            gammas.append(gamma_t)
            #assert (h_t_tr.shape == (self.n_tr, k)), "The shape of h_t_tr needs to be transposed."

            # Update D_t
            alpha_t = 0.5 * np.log((1 + gamma_t) / (1 - gamma_t))
            h_ts.append(alpha_t * h_t_x_l)
            h_ts_test.append(alpha_t * h_t_x_l_test)
            Z_t = np.sqrt(1 - np.square(gamma_t))
            update = np.exp(-alpha_t * np.multiply(Y_train, h_t_x_l)) / Z_t

            D_t = np.multiply(D_t, update)
            D_ts.append(D_t)

            '''
            # List with k classifiers
            h_t = [clf.fit(X_train, Y_train[:, i], sample_weight=D_t[:, i])
                    for i in range(k)]
            # Compute gamma_t according to
            # \sum_{i, l} D_t(i, l) Y_i[l] h_t(x_i, l)
            h_t_x_l = np.array([h_t[i].predict(X_train) for i in range(k)]).T
            h_t_x_l_test = np.array([h_t[i].predict(X_test) for i in range(k)]).T
            gamma_t = np.sum(np.multiply(np.multiply(D_t, Y_train), h_t_x_l))
            gammas.append(gamma_t)

            print(h_t_x_l)

            # Update D_t
            alpha_t = 0.5 * np.log((1 + gamma_t) / (1 - gamma_t))
            print(alpha_t)
            h_ts.append(alpha_t * h_t_x_l)
            h_ts_test.append(alpha_t * h_t_x_l_test)
            Z_t = np.sqrt(1 - np.square(gamma_t))
            update = np.exp(-alpha_t * np.multiply(Y_train, h_t_x_l)) / Z_t
            D_t = np.multiply(D_t, update)
            D_ts.append(D_t)
            '''
        H = sum(h_ts)
        H_test = sum(h_ts_test)
        # Calculate the error of H
        w_init_tr = self._get_init_distr(W_init, raveled, use_train=True)
        w_init_te = self._get_init_distr(W_init, raveled, use_train=False)
        train_error = self._get_ham_loss(w_init_tr, H, Y_train, unravel=True)
        test_error = self._get_ham_loss(w_init_te, H_test, Y_test, unravel=True)

        return (H, H_test, train_error, test_error, gammas, D_ts)
        #return (train_error, test_error, gammas, D_ts)

    # GOOD METHOD, LOOK HERE
    def run_factorized(self, T, weak_learner, W_init, verbose=False):
        # Map instance variables to local variables to be more explicit.
        X_train, X_test = self.X_tr, self.X_te
        Y_train, Y_test = self._one_hot_labels(self.y_tr), self._one_hot_labels(self.y_te)
        n_tr, n_te, k = self.n_tr, self.n_te, self.k
        bias = self.b

        # Compute initial distributions
        raveled = False # False in Factorized Interpretation
        init_d_t_train = True # Use training data
        D_t = self._get_init_distr(W_init, raveled, init_d_t_train, bias)

        h_ts_tr, h_ts_te, gammas, D_ts, vts = [], [], [], [D_t], []
        for t in range(T):
            if verbose:
                print("Round {}".format(t + 1), flush=True)
            # Fit weak learner to data
            alpha_t, v, phi, gamma_t = weak_learner(X_train, Y_train, D_t)
            vts.append(v)
            assert (isinstance(v, np.ndarray)), "Make v a numpy array."
            gammas.append(gamma_t)

            # Use weak learner to make predictions on train and test
            h_t_tr = np.array([alpha_t * phi(X_train[i, :]) * v for i in range(n_tr)])
            h_t_te = np.array([alpha_t * phi(X_test[i, :]) * v for i in range(n_te)])
            h_ts_tr.append(h_t_tr)
            h_ts_te.append(h_t_te)

            # Update D_t
            alpha_t, Z_t = self._get_alpha_and_energy(gamma_t)
            update = np.exp(-alpha_t * np.multiply(Y_train, h_t_tr)) / Z_t
            D_t = np.multiply(D_t, update)
            D_ts.append(D_t)
        H = sum(h_ts_tr)
        H_test = sum(h_ts_te)
        # Calculate the error of H
        # We could make the below cleaner by implementing a _get_error
        # method. It just needs W_init to become an instance variable.
        w_init_tr = self._get_init_distr(W_init, raveled, True, bias)
        w_init_te = self._get_init_distr(W_init, raveled, False, bias)
        train_error = self._get_ham_loss(w_init_tr, H, Y_train, unravel=True)
        test_error = self._get_ham_loss(w_init_te, H_test, Y_test, unravel=True)
        return (train_error, test_error, gammas, D_ts)
