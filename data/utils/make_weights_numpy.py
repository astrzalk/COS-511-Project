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
    #cols_to_use = [0,1,2,3,4,5,6,7,8,9]
    #cols_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    cols_to_use = [0,1]
    df = pd.read_csv(data_path + filename + '.txt', sep=';', usecols = cols_to_use, header=None) 
    np_array = df.values # Convert to numpy array

    np.save(save_path + filename + '.npy', np_array)
    return

def main():
    #weight_path = '../pendigits/single_stump/weights/'
    #weight_path = '../isolet/single_stump/weights/'
    weight_path = '../pendigits_binary/single_stump/weights/'
    for i in range(31):
        save_csv_as_np(weight_path, weight_path, 'weight_{}'.format(i))

if __name__ == "__main__":
    main()

