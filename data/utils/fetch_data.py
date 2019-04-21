#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The code below is a way to extract the Weight matrices
# and vote vectors and return them into a list, so I don't have
# to worry about it in the notebook I am plotting.

import os
import numpy as np

data_dir_pen = "pendigits/single_stump/"
data_dir_iso = "isolet/single_stump/"

def get_array_lists(dir):
    """
    Helper function for get_Wts_and_vts.
    Takes a directory and returns a list of
    numpy arrays from given directory.
    """
    names = sorted([f.name for f in os.scandir(dir)])
    return [np.load(dir + name)for name in names]

def get_Wts_and_vts(rel_path, which_data_set):
    """
    Parameters
    ----------
    which_data_set : String
    If you want the W_ts and v_ts for Pen digits: 'pen',
    if you want instead for isolet: 'iso'.

    Returns
    -------
    ([numpy-array], [numpy-array])
    (Wts, vts), each list should be of length T, which is 30 for both datasets.
    In 'pen': each Wt in Wts should be 7494 x 10, and each v in vts should be 10 x 1.
    In 'iso': each Wt in Wts should be 6238 x 26, and each v in vts should be 26 x 1.
    """
    if which_data_set == 'pen':
        data_dir = rel_path + data_dir_pen
    elif which_data_set == 'iso':
        data_dir = rel_path + data_dir_iso
    # Get Ws
    W_dir = data_dir + 'weights/'
    Wts = get_array_lists(W_dir)
    # Get vs
    v_dir = data_dir + 'vote_vectors/'
    vts = get_array_lists(v_dir)
    return (Wts, vts)

if __name__ == "__main__":
    rel_path = "../"
    # Check fetching of pen data is right
    Wts_pen, vts_pen = get_Wts_and_vts(rel_path, 'pen')
    assert (len(Wts_pen) == 31),\
"Wrong number of rounds for Pen weights. There are {} rounds here.".format(len(Wts_pen))
    assert (len(vts_pen) == 30),\
"Wrong number of rounds for Pen vs. There are {} rounds here.".format(len(vts_pen))

    assert (Wts_pen[5].shape == (7494,10)),\
"Pen weights have wrong shape: {}".format(Wts_pen[5].shape)
    assert (vts_pen[5].shape == (10,1) or vts_pen[5].shape == (10,)),\
"Pen vs have wrong shape: {}".format(vts_pen[5].shape)

    # Check fetching of iso data is right
    Wts_iso, vts_iso = get_Wts_and_vts(rel_path, 'iso')
    assert (len(Wts_iso) == 31),\
"Wrong number of rounds for iso weights. There are {} rounds here.".format(len(Wts_iso))
    assert (len(vts_iso) == 30),\
"Wrong number of rounds for iso vs. There are {} rounds here.".format(len(vts_iso))

    assert (Wts_iso[5].shape == (6238,26)),\
"Iso weights have wrong shape: {}".format(Wts_iso[5].shape)
    assert (vts_iso[5].shape == (26,1) or vts_iso[5].shape == (26,)),\
"Pen vs have wrong shape: {}".format(vts_iso[5].shape)
