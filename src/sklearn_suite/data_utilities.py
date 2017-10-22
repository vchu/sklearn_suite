#!/usr/bin/env python
# Main data utility class that deals with loading and parsing data
#
# This assumes that data was recorded using data_logger_bag and
# loaded using the built in functions from that package
#
#
#

from __future__ import print_function
import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import os
import numpy as np
from pylab import *
import pandas as pd # for nan checking
import cPickle
from random import shuffle
from collections import defaultdict
from data_logger_bag.load_h5_dataset import load_data
from learning_constants import * # imports all of the constants

DEFAULT_SVM_WIN_SIZE = 3
DEFAULT_SVM_GOAL_SIZE = 3

def compute_keys(keys, train_size=0.8, randomize=True):
    '''
    Simple helper that will take a list of keys and split them into
    train and test sets

    Default: will sort the keys randomly for you. Set randomize to False
    if no sorting required

    Input: list of keys (any form), train size (percentage - e.g. 0.8)
    Ouptut: Train/Test keys in list form
    '''

    # TODO: More sophisticated version of making sure duplicates are not
    # Put into both train and test - but also not removed
    # Currently - they are just removed to make an unique set
    keys = np.unique(keys).tolist()

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

def check_keys(data, success_keys, fail_keys):
    '''
    Helper function that looks at the number of keys extracted and number
    of keys requested
    '''

    key_types = [SUCCESS_KEY, FAIL_KEY]
    for sf_key in key_types:
        if len(set(data[DATA_KEY][sf_key].keys())) is not eval('len(set('+sf_key+'_keys))'):
            print('\n################################################################', file=sys.stderr)
            print('WARNING: Number of '+sf_key+' keys does not match extracted runs', file=sys.stderr)
            print('There should be: %d' % eval('len('+sf_key+'_keys)'), file=sys.stderr)
            print('There are: %d\n' % len(data[DATA_KEY][sf_key]) , file=sys.stderr)
            print('################################################################\n', file=sys.stderr)
            import pdb; pdb.set_trace()
        else:
            print('Keys in %s loaded correctly' % sf_key)

def clean_keys(keys):

    key_list = filter(None, keys)

    return key_list

def load_specific_keys_low_mem(data_input, success_keys=None, fail_keys=None, 
                           dir_levels=None, max_level=None, preload=False, 
                           spec_key=False):
    '''
    This function is identical in functionality to load_specific_keys_gen, but
    It loads each database individually and discards afterwards to ensure
    that the computer memory is not overloaded holding each of the
    dictionaries of the entire dataset
    ''' 

    # Create the dictionary to store data
    data = dict()
    data[DATA_KEY] = defaultdict(dict)
    data[FILENAME_KEY] = []
    data[DIRECTORY_KEY] = []

    success_keys = clean_keys(success_keys)
    fail_keys = clean_keys(fail_keys)

    for data_file in data_input:

        # Load the single file - assume not preloaded
        all_data = load_database_segment(data_file, dir_levels=dir_levels, max_level=max_level, spec_key=spec_key)

        # Cycle through all of the files
        data_segment = all_data[0] # Assume we're only loading a single file at a time

        load_specific_keys_single(data_segment, data, data_file=data_file, dir_levels=dir_levels,
                                  success_keys=success_keys, fail_keys=fail_keys, preload=preload)

    # Check if we have all of the keys loaded
    check_keys(data, success_keys, fail_keys)

    return data
    


def load_specific_keys_gen(data_input, success_keys=None, fail_keys=None, 
                           dir_levels=None, max_level=None, preload=False,
                           spec_key=False):
    '''
    This function is intended to be given an h5 file of the form
    data[dir][dir][dir]..[runs] where the directories are
    given by the directory_levels variable.

    spec_key - if True, then the database segment only has ONE key to begin with

    Returns a dictionary of the form
    data[directories] = directory_levels
    data[key] = data[runs]
    data[filename] = filename

    Note: the keys are the EXACT run names

    WARNING: this does not check to make sure there are no
             duplicates in the success and fail key list

    Note: Takes flag (preload) that allows you to pass in a loaded dataset
          to allow for fast indexing of various key sets

    '''

    success_keys = clean_keys(success_keys)
    fail_keys = clean_keys(fail_keys)

    # Check if we had already preloaded it and passed in the data rather than file name
    if not preload:
        all_data = load_database_segment(data_input, dir_levels=dir_levels, max_level=max_level, spec_key=spec_key)
    else:
        all_data = data_input

    # Create the dictionary to store data
    data = dict()
    data[DATA_KEY] = defaultdict(dict)
    data[FILENAME_KEY] = []
    data[DIRECTORY_KEY] = []

    # Cycle through all of the files
    for data_segment in all_data:

        load_specific_keys_single(data_segment, data, data_file=data_input, dir_levels=dir_levels,
                                   success_keys=success_keys, fail_keys=fail_keys, preload=preload)

    # Check if we have all of the keys loaded
    check_keys(data, success_keys, fail_keys)

    return data

def load_specific_keys_single(data_segment, data_store, data_file=None, dir_levels=None,
                              success_keys=None, fail_keys=None, preload=False):

    # Load the data and go through
    if (preload):
        data_store[FILENAME_KEY].append(data_segment[FILENAME_KEY]) # Store away the file we're working on
        data_store[DIRECTORY_KEY].append(data_segment[DIRECTORY_KEY]) # Store away the directory structure
    else:
        # go through the loaded data and split
        filename_val = '_'.join(os.path.split(data_file)[-1].split('.')[0:1])
        data_store[FILENAME_KEY].append(filename_val) # Store away the file we're working on
        data_store[DIRECTORY_KEY].append(dir_levels) # Store away the directory structure

    # Remove the extra data
    del data_segment[FILENAME_KEY]
    del data_segment[DIRECTORY_KEY]

    # Pull out all of the sub dictionaries that directly relate to the given dir_level
    stored_directories = []
    find_directory(data_segment, dir_levels, True, stored_directories)

    # Now cycle through each of the directories to pull out a merged dataset
    for run_dict in stored_directories:
        pull_keys(run_dict, data_store, success_keys, fail_keys)

    # Put it back in
    data_segment[FILENAME_KEY] = data_store[FILENAME_KEY]
    data_segment[DIRECTORY_KEY] = data_store[DIRECTORY_KEY]


def load_database_segment(data_file, dir_levels=None, max_level=None, spec_key=False):
    '''
    Will load multiple data files if needed
    '''

    data_segments = []
    # Check if we have multiple files
    if type(data_file) is list:
        # load multiple data segments
        for dfile in data_file:
            data_segments.append(load_single_database_segment(dfile, dir_levels=dir_levels, max_level=max_level,spec_key=spec_key) )

    else:
        data_segments.append(load_single_database_segment(data_file, dir_levels=dir_levels, max_level=max_level,spec_key=spec_key) )

    return data_segments


def load_single_database_segment(data_file, dir_levels=None, max_level=None, spec_key=False):

    # Load the data and go through
    if data_file.endswith(".h5"):
        all_data = load_data(data_file, '', False, directories=dir_levels, max_level=max_level, spec_key=spec_key) 
    elif data_file.endswith(".pkl"):
        all_data = load_pkl(data_file)
    else:
        print("Error: Wrong file type passed into function.  Not .h5 or .pkl")

    # go through the loaded data and split
    filename_val = '_'.join(os.path.split(data_file)[-1].split('.')[0:1])
    all_data[FILENAME_KEY] = filename_val # Store away the file we're working on
    all_data[DIRECTORY_KEY] = dir_levels # Store away the directory structure

    return all_data


def find_directory(loaded_data, dir_levels, searching, stored_directories):

    # While we still have directories to find
    if searching:
        cur_dir = dir_levels[0]
        if cur_dir in loaded_data:
            find_directory(loaded_data, dir_levels, False, stored_directories)
            #loaded_data = loaded_data[directories.pop(0)] # Pop off the first item 
            
        else:
            # If not found - then recurse down
            for val in loaded_data:
                find_directory(loaded_data[val], dir_levels, True, stored_directories)

    else:
       
        # We found it and now just go down and pull it out 
        directories = list(dir_levels)
        while directories:
            loaded_data = loaded_data[directories.pop(0)] # Pop off the first item 
  
        stored_directories.append(loaded_data)

def pull_keys(loaded_data, data, success_keys, fail_keys):
    '''
    Helper that actually loads the data when passed in the dictionary
    It will store into another dictionary
    '''

    # Now we're at the level we're expecting
    # and where we actually start cycling
    for run_name in loaded_data:

        # Parse the run number to pull out the ones for success and fail
        if run_name in success_keys:
            data[DATA_KEY][SUCCESS_KEY][run_name] = loaded_data[run_name]

        # Check if we're running supervised or unsupervised
        if fail_keys is not None:

            # Parse the run number to pull out the ones for success and fail
            if run_name in fail_keys:
                data[DATA_KEY][FAIL_KEY][run_name] = loaded_data[run_name]


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
        print("Error: Wrong file type passed into function.  Not .h5 or .pkl")

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

def find_nearest_time(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx,array[idx])

def get_segments(data, split_times, time, pad_time=0.5, segment_thresh=1.0, expected_seg_num=None, merge_first=False):
    """
    Helper function that given the times where segments occur, goes through and
    pulls out that from the data.
    Specifically also includes padding time for how much data AFTER the
    split occurs that should be included in the segment
    This allows for the segments to contain information AFTER the segment to fully be able
    to determine the goal has indeed been reach

    For now just keep segments separate (does not include the previous segment's data)

    pad_time: how much padding AFTER the segment do we keep
    segment_thresh: minimum size a segment can be
    expected_seg_num: is passed in will check and throw warning if not the right num
    """

    # Go through the split times and pull out the exact time range and data index
    # Check if the split times include the first split
    if merge_first:
        split_times.pop(0) # remove the first segment number

    if 0.0 not in split_times:
        split_times.insert(0,0.0)

    # Pull out the last time from the data
    last_timestep = time[-1]
    if last_timestep not in split_times:
        split_times.append(last_timestep)

    data_segments = []
    segment_idx = []
    # Now cycle through the split times
    for i in xrange(len(split_times)-1):
        start_time = split_times[i]
        end_time = split_times[i+1]

        # Merge split times if the segment is too small
        if end_time-start_time < segment_thresh:
            continue

        # determine where the true split occurs
        split_idx = find_nearest_time(time, end_time)[0]
        # Now add padding to the end time
        end_time = end_time+pad_time

        # Pull out the closest idx
        s_idx = find_nearest_time(time, start_time)[0]
        e_idx = find_nearest_time(time, end_time)[0]

        # make sure the last index is within the range of the data
        e_idx = min(e_idx,len(data)-1)
        segment_idx.append(split_idx-s_idx)

        # Get the segment value
        data_seg = data[s_idx:e_idx,:]
        data_segments.append(data_seg)

    if expected_seg_num:
        if len(data_segments) != expected_seg_num:
            print('WARNING: segments returned not expected number')
    
    return data_segments, segment_idx

def get_segment_w_idx(data, split_locs):
    # Given data and the lengths of each segment split
    # split the data according to those lengths

    segs = []
    cur_idx = 0
    for i in xrange(len(split_locs)):
        segs.append(data[cur_idx:cur_idx+split_locs[i],:])
        cur_idx += split_locs[i]
    return segs
    

def get_feature_set(segments, idx_loc, window_size=DEFAULT_SVM_WIN_SIZE, goal_size=DEFAULT_SVM_GOAL_SIZE):
    '''
    Given the location within the feature of the segment,
    create training sets that split between pre success and success
    data
    '''

    feat_set = dict()
    for i in xrange(len(segments)):
        # Compute the average change over the feature vector
        # of various time scales
        features = segments[i]
        idx = idx_loc[i]
 
        goal = features[idx-goal_size:idx+goal_size]
        pre_goal = features[0:idx-goal_size]

        # Sample randomly goal_size*2
        pre_goal_idx = np.random.choice(len(pre_goal), goal_size, replace=False)
        pre_goal_idx.sort() # put in order over time
        pre_goal_idx = range(idx-2*goal_size,idx-goal_size)
        pre_goal_vals = [pre_goal[x-window_size:x+window_size] for x in pre_goal_idx]

        # Store the feature set
        if i not in feat_set:
            feat_set[i] = []

        # Compute the mean change over the window size
        for feat in pre_goal_vals:
            if len(feat) < 1:
                import pdb; pdb.set_trace()
            mean_val = np.mean(np.diff(feat,axis=0),axis=0)
            feat_set[i].append((mean_val,FAIL_VAL))

        goal_feat = np.mean(np.diff(goal, axis=0), axis=0)

        feat_set[i].append((goal_feat,SUCCESS_VAL))

    return feat_set

def get_test_feature_set(features, idx, window_size=DEFAULT_SVM_WIN_SIZE, goal_size=DEFAULT_SVM_GOAL_SIZE):
    '''
    Given the location within the feature of the segment,
    create test set that split between pre success and success
    data
    '''

    feat_set = []
    
    # Compute the average change over the feature vector
    # of various time scales
    goal = features[idx-goal_size:idx+goal_size]
    pre_goal = features[0:idx-goal_size]

    # Sample randomly goal_size*2
    pre_goal_idx = np.random.choice(len(pre_goal), goal_size, replace=False)
    pre_goal_idx.sort() # put in order over time
    pre_goal_idx = range(window_size,idx-goal_size)
    pre_goal_vals = [pre_goal[x-window_size:x+window_size] for x in pre_goal_idx]

    # Compute the mean change over the window size
    for feat in pre_goal_vals:
        if len(feat) < 1:
            import pdb; pdb.set_trace()
        mean_val = np.mean(np.diff(feat,axis=0),axis=0)
        feat_set.append((mean_val,FAIL_VAL))

    # This is the same as in the train set
    goal_feat = np.mean(np.diff(goal, axis=0), axis=0)

    feat_set.append((goal_feat,SUCCESS_VAL))

    return feat_set


def mean_change_value(value, window_size=2):

    import pdb; pdb.set_trace()
    mean_vec = []
    for i in xrange(len(value)-window_size):
        val = value[i:i+window_size]
        diff_mean = np.mean(np.diff(val, axis=0),axis=0)
        mean_vec.append(diff_mean)

    return mean_vec
