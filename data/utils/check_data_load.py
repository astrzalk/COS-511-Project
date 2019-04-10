#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Data should look like the following
# pendigits.tra	Training	7494
# pendigits.tes	Testing		3498
# 16 features, 1 class col

if __name__ == "__main__":
    X_train = np.load('../pendigits_train_data.npy')
    y_train = np.load('../pendigits_train_labels.npy')
    print("The size of the training data is {}\
 with corresponding labels having shape {}.".format(X_train.shape, y_train.shape))
    X_test = np.load('../pendigits_test_data.npy')
    y_test = np.load('../pendigits_test_labels.npy')
    print("The size of the test data is {}\
 with corresponding labels having shape {}.".format(X_test.shape, y_test.shape))
