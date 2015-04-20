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

# Import specific constants to project here:
from iros2015_constants import all_object_keys    
from default_features import *

def load_train_test_pkl(pkl_file, set_name):
    '''
    Note: This function should be replaced whenever the data
    and domain changes
    '''
    # parse the filename and determine what object we're working with
    filename = os.path.split(pkl_file)[-1]
    file_key = filename.split('.')[0]

    # Pull out the training keys we want
    success_keys = all_object_keys[file_key][TRAIN_KEY][set_name][SUCCESS_KEY]
    fail_keys = all_object_keys[file_key][TRAIN_KEY][set_name][FAIL_KEY]
    train_data = load_specific_keys(pkl_file, success_keys, fail_keys)

    # Pull out the testing keys we want
    success_keys = all_object_keys[file_key][TEST_KEY][SUCCESS_KEY]
    fail_keys = all_object_keys[file_key][TEST_KEY][FAIL_KEY]
    test_data = load_specific_keys(pkl_file, success_keys, fail_keys)

    return (train_data, test_data)

def train_single_pkl(data, set_name):

    # Actually train?
    compute_features(test)


def get_features(data):

    compute_features(data, skip_topics=[], state_name='social_affordance_state', ft_norm=False)




def affordance_learning(objects, actions, train_types, data_file):

    # Specific to affordances
    for obj in objects:
        for act in actions:
            affordance = '_'.join((obj,act)) 
            print "Computing affordance %s:" % affordance 

            # Cycle through each type 
            for train_type in train_types:
                print "Training set %s" % train_type
                
                # Pull out trian and test data
                (train_data, test_data) = load_train_test(act, obj, train_type, data_file)

                # Pull out features
                get_features(train_data)
                #train_single_pkl(pkl_file, train_type)



def load_train_test(act, obj, set_type, data_file):

    affordance = '_'.join((obj,act))
    directories = ['bag_files', set_type, obj, act]
    # Pull out the keys that we need to load the dataset
    success_keys = all_object_keys[affordance][TRAIN_KEY][set_type][SUCCESS_KEY]
    fail_keys = all_object_keys[affordance][TRAIN_KEY][set_type][FAIL_KEY]
       
    # Load datafile
    train_data = load_specific_keys_gen(data_file, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, max_level=3)

    # Pull out the testing keys we want
    success_keys = all_object_keys[affordance][TEST_KEY][SUCCESS_KEY]
    fail_keys = all_object_keys[affordance][TEST_KEY][FAIL_KEY]
    test_data = load_specific_keys_gen(data_file, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, max_level=3)

    return (train_data, test_data)

def main():

    ####################################
    # Setup the parser and add arguments
    #####################################
    parser = argparse.ArgumentParser(description='Template scikit supervised training program') 

    # Input files
    parser.add_argument('-d', action='store', dest='data_dir', help='Location to directory of h5 or pkl files', default=None)
    parser.add_argument('-i', action='store', dest='data_file', help='Location to file in h5 or pkl form', default=None)
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
    if results.data_dir is None and results.data_file is None:
        parser.print_help()
        raise Exception("Error: no input file/directory given")

    # Check if we're doing affordance learning
    if results.objects:
        if results.data_dir == None:
            affordance_learning(results.objects, results.actions, results.train_types, results.data_file)
        else:
            affordance_learning_dir(results.objects, results.actions, results.train_types, results.data_dir)




if __name__== "__main__":
    main()
    print "done"




'''
OLD
def affordance_learning_dir(objects, actions, train_types, data_dir):

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
                    
                    # Pull out trian and test data
                    (train_data, test_data) = load_train_test_pkl(pkl_file, train_type)

                    # Pull out features
                    get_features(train_data)
                    #train_single_pkl(pkl_file, train_type)
            else:
                print "Affordance: %s doesn't exist" % pkl_file
'''
