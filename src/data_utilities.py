#!/usr/bin/env python
# Main data utility class that deals with loading and parsing data
#
# This assumes that data was recorded using data_logger_bag and
# loaded using the built in functions from that package
#
#
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import os
import numpy as np
import pandas as pd # for nan checking
import cPickle
from random import shuffle
from load_h5_dataset import load_data
from learning_constants import * # imports all of the constants


def compute_keys(keys, train_size=0.8, randomize=True):
    '''
    Simple helper that will take a list of keys and split them into
    train and test sets

    Default: will sort the keys randomly for you. Set randomize to False
    if no sorting required

    Input: list of keys (any form), train size (percentage - e.g. 0.8)
    Ouptut: Train/Test keys in list form
    '''

    # Create keys
    train_size_percentage = train_size
    total_runs = len(keys)
    train_size = int(math.floor(total_runs * train_size_percentage))

    # Will shuffle the keys if flag set
    if (randomize):
        shuffle(keys)
        shuffle(keys)
    
    train_keys = keys[0:train_size]
    test_keys = keys[train_size:total_runs]

    return (train_keys, test_keys)


def load_pkl(path_name):
    '''
    Simple helper function that loads a pickle file given a path to it

    input: path to pkl file
    output: loaded pickle file

    '''

    # Load pickle file
    with open(path_name, "r") as file_path:
       loaded_file = cPickle.load(file_path)

    return loaded_file

def save_pkl(path_name, data):
    '''
    Simple helper that writes data to pkl file

    Input: data 
           path_name - location to and the filename of data
    '''

    cPickle.dump(data, open(path_name, "w"), cPickle.HIGHEST_PROTOCOL)

def check_data(single_run):
    '''
    Function that checks a single run and removes nans.
    It will directly replace the nans with the closest value to the nan value

    input: single run for one interaction in the form of a <>
    output: same run with nans replaced with closest value
    '''

    if single_run.ndim > 1:
        nan_idx = np.where(np.any(pd.isnull(single_run), axis=1))[0]
    else:
        nan_idx = np.where(pd.isnull(single_run))[0]

    if len(nan_idx) > 0:

        # Go through the values that have NaNs and replace with ones closest
        nan_split_idx_arr = nan_idx[np.where(np.diff(nan_idx) > 1)[0]]+1
        if len(nan_split_idx_arr) > 0:
            cur_nan_idx = 0
            for nan_split_idx in nan_split_idx_arr:
                new_idx = min(max(0,nan_split_idx),np.shape(single_run)[0])
                single_run[nan_idx[cur_nan_idx:nan_split_idx]] = single_run[new_idx]
                cur_nan_idx = nan_split_idx

            single_run[nan_idx[cur_nan_idx::]] = single_run[cur_nan_idx]
        else:
            new_idx = min(max(0,amax(nan_idx)+1),np.shape(single_run)[0])
            single_run[nan_idx] = single_run[new_idx]

    return single_run


def load_specific_keys(data_file, success_keys, fail_keys):
    '''
    This function is intended to be given a pickle file of the form
    data[merged][object][action][runs]

    Returns a dictionary of the form
    data[positive_key] = data[runs]
    data[negative_key] = data[runs]
    data[filename] = filename
    data[object] = object
    data[action] = action

    Note: the success_keys and fail_keys are the EXACT run
          names

    WARNING: this does not check to make sure there are no
             duplicates in the success and fail key list
    '''

    # Create the dictionary to store split data (neg/pos examples)
    split_data = dict()
    split_data[SUCCESS_KEY] = dict()
    split_data[FAIL_KEY] = dict()

    # Load the data and go through
    if data_file.endswith(".h5"):
        all_data = load_data(data_file, '', False) 
    elif data_file.endswith(".pkl"):
        all_data = load_pkl(data_file)
    else:
        print "Error: Wrong file type passed into function.  Not .h5 or .pkl"

    # go through the loaded data and split
    filename_val = '_'.join(os.path.split(data_file)[-1].split('.')[0:1])
    split_data[FILENAME_KEY] = filename_val # Store away the file we're working on

    for file_name in all_data:
        file_data = all_data[file_name]

        for object_name in file_data:
            data = file_data[object_name]
            split_data[OBJECT_KEY] = object_name

            for action_name in data:
                data_store = data[action_name]
                split_data[ACTION_KEY] = action_name

                # This is where we actually start cycling
                for run_name in data_store:

                    # Parse the run number to pull out the ones for success and fail
                    if run_name in success_keys:
                        split_data[SUCCESS_KEY][run_name] = data_store[run_name]

                    if run_name in fail_keys:
                        split_data[FAIL_KEY][run_name] = data_store[run_name]

    return split_data




