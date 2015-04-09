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
OBJECT_KEY = 'object'
ACTION_KEY = 'action'
RUN_KEY = 'numbers'
NAME_KEY = 'names'


####################################################
# Constants for training and testing
####################################################
SUCCESS_KEY = 'success'
FAIL_KEY = 'fail'
FEAT_KEY = 'features'
LABEL_KEY = 'labels'
RUN_KEY = 'ID'
CV_KEY = 'grid'
CLF_KEY = 'clf'
TRAIN_KEY = 'train'
TEST_KEY = 'test'
ALL_KEY = 'all'
FEATURE_LOC = 0
LABEL_LOC = 1

####################################################
# Constants that are feature specific
####################################################
OBJECT_CLUSTER_FEAT_PREFIX = 'object_'
COMPACT_NORM = 1e-6
RGBA_COLOR_NORM = 255 




