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
from evaluation_utilities import *

# Import specific constants to project here:
from iros2015_constants import all_object_keys, USER_KEY, ALL_AFFORDANCES, AFF_FEATURES, OBJ_FEATURES, SOCIAL_STATE
from default_features import *

def affordance_learning(objects, actions, train_types, data_file):
    '''
    This function is very specific to affordance learning
    It calls generic functions throughout this and other utility files
    '''
    
    # Specific to affordances
    for obj in objects:
        for act in actions:
            affordance = '_'.join((obj,act)) 
            
            # Check if the combination actually exists
            if affordance not in ALL_AFFORDANCES:
                continue

            print "\nComputing affordance %s:\n" % affordance 

            # Cycle through each type  (self, guided, user)
            for train_type in train_types:

                print "Training set %s\n" % train_type

                #################################################### 
                # Pull out data and create features
                ####################################################
                # Create necessary information to extract data from h5
                affordance = '_'.join((obj,act))

                # Specify the depth (optional - entirely for speedup purposes)
                if train_type == USER_KEY:
                    train_dir_depth = 4
                    directories = [obj, act]
                else:
                    train_dir_depth = 3
                    directories = ['bag_files', train_type, obj, act]

                # Pull out the keys that we need to load the dataset
                success_keys = all_object_keys[affordance][TRAIN_KEY][train_type][SUCCESS_KEY]
                fail_keys = all_object_keys[affordance][TRAIN_KEY][train_type][FAIL_KEY]
                
                # Pull out train data
                train_data = load_specific_keys_gen(data_file, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, max_level=train_dir_depth)

                # Pull out features
                print "Training with these features:"
                print AFF_FEATURES
                print OBJ_FEATURES
                (train_feat, train_keys) = get_features(train_data, AFF_FEATURES, OBJ_FEATURES, state_name=SOCIAL_STATE)
                ####################################################
                # Train each model and save away
                ####################################################

                success_hmm = train_hmm_gridsearch(train_feat[FEAT_KEY][SUCCESS_KEY], cv=5, n_jobs=5)
                title = affordance+'_'+train_type+ '_success.pkl'
                cPickle.dump(success_hmm, open(os.path.join('hmms',title), "w"), cPickle.HIGHEST_PROTOCOL)

                fail_hmm = train_hmm_gridsearch(train_feat[FEAT_KEY][FAIL_KEY], cv=5, n_jobs=5)
                title = affordance+'_'+train_type+'_fail.pkl'
                cPickle.dump(fail_hmm, open(os.path.join('hmms',title), "w"), cPickle.HIGHEST_PROTOCOL)

def main():

    #####################################
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
    if results.data_file is None:
        parser.print_help()
        raise Exception("Error: no input file/directory given")

    # Check if we're doing affordance learning
    if results.objects:
        affordance_learning(results.objects, results.actions, results.train_types, results.data_file)




if __name__== "__main__":
    main()
    print "done"

