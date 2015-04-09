#!/usr/bin/env python
# Template for getting data loaded and chugging through
#
#
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import argparse
import os
from learning_constants import *
from learning_utilities import *
from data_utilities import *

def load_data(pkl_file, set_name):
    '''
    Note: This function should be replaced whenever the data
    and domain changes
    '''
    # Just for testing - this will only be added specific to file
    from iros2015_constants import all_object_keys    

    # parse the filename and determine what object we're working with
    filename = os.path.split(pkl_file)[-1]
    file_key = filename.split('.')[0]

    # Pull out the training keys we want
    success_keys = all_object_keys[file_key][TRAIN_KEY][set_name][SUCCESS_KEY]
    fail_keys = all_object_keys[file_key][TRAIN_KEY][set_name][FAIL_KEY]
    training_data = load_specific_keys(pkl_file, success_keys, fail_keys)

    # Pull out the testing keys we want
    success_keys = all_object_keys[file_key][TEST_KEY][SUCCESS_KEY]
    fail_keys = all_object_keys[file_key][TEST_KEY][FAIL_KEY]
    test_data = load_specific_keys(pkl_file, success_keys, fail_keys)


def train_single_pkl(pkl_file, set_name):

    load_data(pkl_file, set_name)




def affordance_learning(objects, actions, train_types, data_dir):

    # Specific to affordances
    for obj in objects:
        for act in actions:
            affordance = '_'.join((obj,act)) 
            pkl_file = os.path.join(data_dir, affordance+'.pkl')

            # Check if combination exists
            if(os.path.isfile(pkl_file)):
                print "Computing pkl file %s:" % pkl_file

                # Cycle through each type 
                for train_type in train_types:
                    print "Training set %s" % train_type
                    train_single_pkl(pkl_file, train_type)
            else:
                print "Affordance: %s doesn't exist" % pkl_file


def main():

    ####################################
    # Setup the parser and add arguments
    #####################################
    parser = argparse.ArgumentParser(description='Template scikit supervised training program') 

    # Input files
    parser.add_argument('-i', action='store', dest='data_dir', help='Location to file in h5 or pkl form', default=None)
    parser.add_argument('-t', action='append', dest='train_types', help='Custom string for labeling the training form. e.g. self, guided', default=[])

    # Specific to affordances
    parser.add_argument('-O', action='append', dest='objects', help='Add object to train over', default=[])
    parser.add_argument('-A', action='append', dest='actions', help='Add action to train over', default=[])

    # Retreive arguments
    results = parser.parse_args()

    #########################################
    # Start processing the arguments
    #########################################

    # Check if we were given a file
    if results.data_dir is None:
        parser.print_help()
        raise Exception("Error: no input file given")

    # Check if we're doing affordance learning
    if results.objects:
        affordance_learning(results.objects, results.actions, results.train_types, results.data_dir)





if __name__== "__main__":
    main()
    print "done"
