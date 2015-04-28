#!/usr/bin/env python
#
# This file contains many of the constants used throughout the package
#
# Author: Vivian Chu (vchu@gatech.edu)

import roslib; roslib.load_manifest("sklearn_suite")
import rospy

#####################################################
# Constants for Data Loading
#####################################################
FILENAME_KEY = 'filename'
DIRECTORY_KEY = 'directory_level'
OBJECT_KEY = 'object'
ACTION_KEY = 'action'
RUN_KEY = 'numbers'
NAME_KEY = 'names'


####################################################
# Constants for training and testing
####################################################
DATA_KEY = 'data'
SUCCESS_KEY = 'success'
FAIL_KEY = 'fail'
FEAT_DICT_KEY = 'feat_dict'
FEAT_KEY = 'features'
MERGED_FEAT = 'merged_features'
MERGED_FEAT_KEYS = 'merged_feat_keys'
LABEL_KEY = 'labels'
RUN_KEY = 'ID'
CV_KEY = 'grid'
CLF_KEY = 'clf'
TRAIN_KEY = 'train'
TEST_KEY = 'test'
ALL_KEY = 'all'
FEATURE_LOC = 0
LABEL_LOC = 1
SUCCESS_VAL = 0
FAIL_VAL = 1

# For evaluation
PREDICTY_KEY = 'predict_Y'
TESTY_KEY = 'test_Y'
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
M_PRECISION = 'mean_precision'
M_RECALL = 'mean_recall'
M_F1 = 'mean_f1'

####################################################
# Constants that are feature specific
####################################################
OBJECT_CLUSTER_FEAT_PREFIX = 'object_'
COMPACT_NORM = 1e-6
RGBA_COLOR_NORM = 255 




