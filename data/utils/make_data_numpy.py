#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv


def one_hot_labels(y_s, k):
        """
        y: N by 1 numpy-array corresponding to the labels.
        k: float, corresponding to number of unique classes.
        returns: Y, an N by k numpy-array with each row containing
                 a 1 for the correct class and -1 otherwise.
        """
        N = len(y_s)
        Y = np.ones((N, k)) * -1
        for i in range(N):
            for l in y_s[i]:
                Y[i, l] *= -1 # Make the correct class become +1.
        return Y

def save_csv_as_np(data_path, save_path, filename, k):
    """
    Takes a path where the data is (in a csv file format),
    converts file to numpy array, and saves numpy array
    into two files data and labels with the given filename.

    :returns: None, two saved .npy files at save_path
    """
    x_s = []
    y_s = []
    with open(data_path) as txt:
        data = txt.readlines()
        for row in data:
            str_row = row.split(';')
            x = str_row[0].split(',')
            x = [float(i) for i in x]
            y = str_row[1].split(',')
            y = [int(i) for i in y]
            x_s.append(x)
            y_s.append(y)

    np_x_s = np.array(x_s)
    np_y_s = one_hot_labels(y_s, k)

    #if 'train' in filename:
        #assert(np_array.shape[0] == 7494)
        #assert(np_array.shape[0] == 6238)
    #else:
        #assert(np_array.shape[0] == 3498)
        #assert(np_array.shape[0] == 1559)

    np.save(save_path + filename + '_data.npy', np_x_s)
    np.save(save_path + filename + '_labels.npy', np_y_s)
    return

def main():
    #data_path_train = '../pendigits.tra'
    #data_path_test = '../pendigits.tes'
    #save_csv_as_np(data_path_train, '../', 'pendigits_train')
    #save_csv_as_np(data_path_test, '../', 'pendigits_test')
    #data_path_train = '../isolet/isolet.tra'
    #data_path_test = '../isolet/isolet.tes'
    #save_csv_as_np(data_path_train, '../isolet/', 'isolet_train')
    #save_csv_as_np(data_path_test, '../isolet/', 'isolet_test')

    # Assume that y constitutes integers that start at 0
    k = 10
    data_path = '../pendigits/'
    data_path_train = data_path + 'train.tra'
    data_path_test = data_path + 'test.tes'
    save_csv_as_np(data_path_train, data_path, 'train', k)
    save_csv_as_np(data_path_test, data_path, 'test', k)



if __name__ == "__main__":
# Data should look like the following
# pendigits.tra	Training	7494
# pendigits.tes	Testing		3498
# 16 features, 1 class col
    main()

