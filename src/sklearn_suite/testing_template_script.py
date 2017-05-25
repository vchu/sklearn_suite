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
from constants.global_key_storage import *
from constants.HRI2016_constants import train_test_set, test_set_only, ALL_AFFORDANCES
from default_features import *

def extract_affordance_database(database, filename):

    # pull out affordance to load the directory structure
    affordance_list = filename.split('_')[0:2]
    directories = affordance_list

    # Load the database from the h5 set
    all_data = load_database_segment(database, dir_levels=directories, max_level=4)

    return all_data

def extract_affordance_test_set(databases, filename, test_set_num, split_type=PERCENTAGE_SPLIT):

    # pull out affordance to load the directory structure
    affordance_list = filename.split('_')[0:2]
    directories = affordance_list
    affordance = '_'.join(affordance_list)

    # pull out affordance
    success_keys = test_set_only[affordance][test_set_num][split_type][SUCCESS_KEY]
    fail_keys = test_set_only[affordance][test_set_num][split_type][FAIL_KEY]

    # Load the specific keys
    test_key_data = load_specific_keys_low_mem(databases, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, preload=True)

    # Pull out test features
    (test_feat, test_keys) = get_features(test_key_data, AFF_FEATURES['FT_EEF'], AFF_FEATURES['visual'], state_name=SOCIAL_STATE)

    return test_feat


def testing_classifier(databases, directory, preload=False, split_type=PERCENTAGE_SPLIT):

    done_files = []
    results = dict() # Store the results of the testing

    # Get all files
    all_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    # Go through for each affordance..
    for affordance in ALL_AFFORDANCES:

        print '\n####################################'
        print 'Testing affordance: %s' % affordance
        print '####################################'

        idx = [i for i, s in enumerate(all_files) if affordance in s]
        aff_files = [all_files[i] for i in idx]

        # Check if we have any files for that affordance
        if len(aff_files) > 1:
            ##########################################
            # Load the data to test features on
            ##########################################

            if preload:
                test_feats = load_pkl(databases)
            else:
                print '\nLoading test dataset'
                test_set = 0 
                test_feats = extract_affordance_test_set(databases, affordance, test_set, split_type=split_type)

        else:
            print 'No trained files for affordance: %s. Going to next affordance' % affordance
            continue

        # Go through each pkl file (trained classifier) and test
        for f in aff_files:

            ##########################################
            # Load the classifier dictionary
            ##########################################

            # Pull out the file specific names
            filename = f.split('.pkl')[0]
            filename_split = filename.split('_')

            # Check if we're dealing with a success/fail set
            if filename_split[-1] in 'success_fail':
                filename_specific = '_'.join(filename_split[0:-1])

                # Skip file if we've done that pair already
                if filename_specific in done_files:
                    continue

                print '\nEvaluating file: %s' % filename_specific

                # Load up the dictionary
                learning_dict = dict()
                learning_dict[SUCCESS_KEY] = load_pkl(os.path.join(directory, filename_specific+'_success.pkl'))
                learning_dict[FAIL_KEY] = load_pkl(os.path.join(directory, filename_specific+'_fail.pkl'))
       
                # Store away we've done this pair 
                done_files.append(filename_specific)

            else: # just load the whole file
                filename_specific = '_'.join(filename_split[0::])
                learning_dict = load_pkl(os.path.join(directory, f))

             
            ##########################################
            # Test the classifier
            ##########################################

            # Specifically test HMMs
            predict_Y, test_Y = test_hmm(test_feats[MERGED_FEAT], learning_dict)        

            # Store with key to later pull out
            results[filename_specific] = defaultdict(dict)
            results[filename_specific][test_set][PREDICTY_KEY] = predict_Y
            results[filename_specific][test_set][TESTY_KEY] = test_Y

        # Write away the file for storage?
        save_pkl(os.path.join(directory, split_type+'_results.pkl'), results)

    
def print_results(results_file): 

    # Check if file exists
    if os.path.isfile(results_file) and results_file.endswith(".pkl"):
        # Load the results file
        results = load_pkl(results_file)    
    else: 
        print ("Warning: Not a valid pkl file")
        sys.exit()
    
    ##########################################
    # Evaluate scores
    ##########################################

    # For all of the files now print out uniformly
    scores = compute_results(results)
    print_affordance_latex(scores)
    print_avg_results_latex(scores)


def main():

    #####################################
    # Setup the parser and add arguments
    #####################################
    parser = argparse.ArgumentParser(description='Template scikit supervised training program') 

    # Input files
    parser.add_argument('-d', action='store', dest='trained_dir', help='Location to directory of trained pkl files', default=None)
    parser.add_argument('-i', action='append', dest='data_file', help='Location to data in h5 format', default=[])
    parser.add_argument('-b', action='store', dest='database_dir', help='Location to directory of databases', default=None)
    parser.add_argument('-f', action='store', dest='results_file', help='Location to computed results in pkl format', default=None)
    parser.add_argument('-t', action='store', dest='test_feat', help='Location to test features in pkl format', default=None)
    parser.add_argument('-s', action='store', dest='split_type', help='How are we splitting the data', default=PERCENTAGE_SPLIT)

    # Retreive arguments
    results = parser.parse_args()

    #########################################
    # Start processing the arguments
    #########################################

    if results.results_file is not None:
        print_results(results.results_file)

    else:

        if results.trained_dir is not None:
            if results.test_feat is not None:
                testing_classifier(results.test_feat, results.trained_dir, preload=True)

            else:
                if len(results.data_file) < 1:
                    if results.database_dir is None: 
                        parser.print_help()
                        raise Exception("Error: no input training directory or h5 database given")
                    else:

                        # Load the files in the database_dir and pass in
                        database_files = [os.path.join(results.database_dir,f) for f in os.listdir(results.database_dir)]
                        testing_classifier(database_files, results.trained_dir, split_type=results.split_type)
                
                else:
                    testing_classifier(results.data_file, results.trained_dir, split_type=results.split_type)



if __name__== "__main__":
    main()
    print "done"


