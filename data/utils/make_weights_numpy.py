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
    cols_to_use = [0,1,2,3,4,5,6,7,8,9]
    df = pd.read_csv(data_path + filename + '.txt', sep=';', usecols = cols_to_use, header=None) 
    #df = pd.read_csv(data_path + filename + '.txt', sep=';', header=None) 
    np_array = df.values # Convert to numpy array

    #print(np_array[0,:])
    #print(np_array[7493,:])
    #assert(np_array.shape[0] == 7494), "{}".format(np_array.shape[0])
    #print(np_array.shape[1])
    #assert(np_array.shape[1] == 10)
    #assert 0
    np.save(save_path + filename + '.npy', np_array)
    return

def main():
    weight_path = '../pendigit_weights_singlestump/'
    save_csv_as_np(weight_path, weight_path, 'weight_1')
    save_csv_as_np(weight_path, weight_path, 'weight_2')
    save_csv_as_np(weight_path, weight_path, 'weight_3')
    save_csv_as_np(weight_path, weight_path, 'weight_4')
    save_csv_as_np(weight_path, weight_path, 'weight_5')

if __name__ == "__main__":
    main()

