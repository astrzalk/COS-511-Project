#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def save_csv_as_np(data_path, save_path, filename):
    """
    Takes a path where the data is (in a csv file format),
    converts file to numpy array, and saves numpy array
    into two files data and labels with the given filename.

    :returns: None, two saved .npy files at save_path
    """
    df = pd.read_csv(data_path, header=None)
    np_array = df.values # Convert to numpy array
    #if 'train' in filename:
        #assert(np_array.shape[0] == 7494)
        #assert(np_array.shape[0] == 6238)
    #else:
        #assert(np_array.shape[0] == 3498)
        #assert(np_array.shape[0] == 1559)

    np.save(save_path + filename + '_data.npy', np_array[:, :-1])
    np.save(save_path + filename + '_labels.npy', np_array[:, -1])
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

    data_path_train = '../synthHalf/train.tra'
    data_path_test = '../synthHalf/test.tes'
    save_csv_as_np(data_path_train, '../synthHalf/', 'train')
    save_csv_as_np(data_path_test, '../synthHalf/', 'test')



if __name__ == "__main__":
# Data should look like the following
# pendigits.tra	Training	7494
# pendigits.tes	Testing		3498
# 16 features, 1 class col
    main()

