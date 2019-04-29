#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arff
import numpy as np
import shutil

# TODO: Reimplement later with better numpy vectorized code
def get_good_data(filename):
    arr = [[ele for ele in row] for row in arff.load(filename)]
    # arr = [[float(ele) for ele in row] for row in filename]
    np_arr = np.array(arr)
    for i in range(np_arr.shape[0]):
        if np_arr[i, -1] > '5':
            np_arr[i, -1] = '1'
        else:
            np_arr[i, -1] = '0'
    return np_arr

if __name__ == "__main__":
    # with open('pendigitsTrain.arff', 'r') as f:
    #     data_flag = False
    #     for line in f:
    #         if data_flag:
    #             line[-2]
    #         if line == "@DATA\n":
    #             data_flag = True

    filename = "pendigitsTest"
    with open(filename + "binary" + ".arff", "w") as f_new, open(filename + ".arff", "r") as f_old:
        data_flag = False
        for line in f_old:
            new_line = line
            if data_flag:
                if int(line[-2]) > 5:
                    l = new_line.split(",")
                    l[-1] = '1\n'
                    new_line = ','.join(l)
                else:
                    l = new_line.split(",")
                    l[-1] = '0\n'
                    new_line = ','.join(l)
            if line == "@DATA\n":
                data_flag = True
            f_new.write(new_line)


    # name_tr = "pendigitsTrain.arff"
    # name_te = "pendigitsTest.arff"
    # data_train = get_good_data(name_tr)
    # data_test =  get_good_data(name_te)
    # np.savetxt("pen_train.txt", data_train, delimiter=",")
    # np.savetxt("pen_test.txt", data_test, delimiter=",")

