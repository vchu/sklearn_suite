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
from mpl_toolkits.mplot3d import Axes3D


# Import specific constants to project here:
from iros2015_constants import all_object_keys, USER_KEY, SELF_KEY, GUIDED_KEY
from iros2015_constants import ALL_AFFORDANCES, AFF_FEATURES, OBJ_FEATURES, SOCIAL_STATE
from default_features import *

def affordance_plotting(objects, actions, train_types, data_file):
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

            afford_feat = defaultdict(dict)
            # Cycle through each type  (self, guided, user)
            for train_type in train_types:

                print "Extracting feature set %s\n" % train_type

                #################################################### 
                # Pull out data and create features
                ####################################################
                # Create necessary information to extract data from h5
                affordance = '_'.join((obj,act))

                # Specify the depth (optional - entirely for speedup purposes)
                directories = [obj, act]

                # Pull out the keys that we need to load the dataset
                success_keys = all_object_keys[affordance][TRAIN_KEY][train_type][SUCCESS_KEY]
                fail_keys = all_object_keys[affordance][TRAIN_KEY][train_type][FAIL_KEY]
               
                # Pull out database section
                data_segment = load_database_segment(data_file, dir_levels=directories, max_level=4)
                
                # Pull out train data
                train_data = load_specific_keys_gen(data_segment, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, preload=True)

                # Pull out features
                (train_feat, train_keys) = get_features(train_data, AFF_FEATURES, OBJ_FEATURES, state_name=SOCIAL_STATE)

                test_set_num = 0
                # Pull out test data
                success_keys = all_object_keys[affordance][TEST_KEY][SUCCESS_KEY][test_set_num]
                fail_keys = all_object_keys[affordance][TEST_KEY][FAIL_KEY][test_set_num]

                # Load the specific keys
                test_key_data = load_specific_keys_gen(data_segment, success_keys=success_keys, fail_keys=fail_keys, dir_levels=directories, preload=True) 

                # Pull out features
                (test_feat, test_keys) = get_features(test_key_data, AFF_FEATURES, OBJ_FEATURES, state_name=SOCIAL_STATE)
                afford_feat[train_type][TRAIN_KEY] = train_feat
                afford_feat[train_type][TEST_KEY] = test_feat
    
            # Store features to file for easy plotting
            save_pkl(os.path.join('plotting',affordance+'.pkl'), afford_feat)


def plot_affordance(pkl_file):

    features = load_pkl(pkl_file)
    affordance = pkl_file.split('.pkl')[0]

    markers = ['o','v','s']
    colors = ['r','g','b']
    # Create figure that is 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(affordance)

    train_test_key = TRAIN_KEY

    # Pull out features for each type
    for train_type in features:
        if train_type == USER_KEY:
            m = markers[0]
            c = colors[0]
        elif train_type == SELF_KEY:
            m = markers[1]
            c = colors[1]
        else:
            m = markers[2]
            c = colors[2]

        # pull out feat_dict
        feat_dict = features[train_type][train_test_key][FEAT_DICT_KEY]

        success_end_points = []
        # pull out the EEF pos rel to obj for each run
        for run_key in feat_dict[SUCCESS_KEY]:
            success_end_points.append(feat_dict[SUCCESS_KEY][run_key]['left_EEF_obj'][-1])

        fail_end_points = []
        for run_key in feat_dict[FAIL_KEY]:
            fail_end_points.append(feat_dict[FAIL_KEY][run_key]['left_EEF_obj'][-1])

        spts = np.vstack(success_end_points)
        fpts = np.vstack(fail_end_points)

        ax.scatter(spts[:,0],spts[:,1],spts[:,2], c=c, marker='o', s=80)
        ax.scatter(fpts[:,0],fpts[:,1],fpts[:,2], c=c, marker='x', s=80)

    # Create a legend
    line1 = Line2D(range(1), range(1), color="white", marker='o', markersize=10, markerfacecolor="white")
    line2 = Line2D(range(1), range(1), color="black", marker='x', markersize=10, markerfacecolor="black")
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    red_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    x_proxy = plt.Circle((0,0), fc='g')
    ax.legend([blue_proxy,red_proxy, green_proxy, line1, line2],['Guided','User','Self', 'Success', 'Fail'])
    '''
    # Pull out test points and plot
    feat_dict = features[train_type][TEST_KEY][FEAT_DICT_KEY]

    success_end_points = []
    # pull out the EEF pos rel to obj for each run
    for run_key in feat_dict[SUCCESS_KEY]:
        success_end_points.append(feat_dict[SUCCESS_KEY][run_key]['left_EEF_obj'][-1])

    fail_end_points = []
    for run_key in feat_dict[FAIL_KEY]:
        fail_end_points.append(feat_dict[FAIL_KEY][run_key]['left_EEF_obj'][-1])

    spts = np.vstack(success_end_points)
    fpts = np.vstack(fail_end_points)

    ax.scatter(spts[:,0],spts[:,1],spts[:,2], c='m', marker='o', s=80)
    ax.scatter(fpts[:,0],fpts[:,1],fpts[:,2], c='m', marker='x', s=80)
    '''
    import pdb; pdb.set_trace()
    savefig(affordance+'.png', bbox_inches='tight')


def main():

    #####################################
    # Setup the parser and add arguments
    #####################################
    parser = argparse.ArgumentParser(description='Template scikit supervised training program') 

    # Input files
    parser.add_argument('-d', action='store', dest='data_dir', help='Location to directory of h5 or pkl files', default=None)
    parser.add_argument('-i', action='store', dest='data_file', help='Location to file in h5 or pkl form', default=None)
    parser.add_argument('-t', action='append', dest='train_types', help='Custom string for labeling the training form. e.g. self, guided', default=[])
    parser.add_argument('-f', action='store', dest='plot_file', help='Location to pkl file of features to play with', default=None)

    # Specific to affordances
    parser.add_argument('-O', action='append', dest='objects', help='Add object to train over', default=[])
    parser.add_argument('-A', action='append', dest='actions', help='Add action to train over', default=[])

    # Retreive arguments
    results = parser.parse_args()

    #########################################
    # Start processing the arguments
    #########################################

    if results.plot_file is not None:
    
        plot_affordance(results.plot_file)
    else:

        # Check if we were given a file
        if results.data_file is None:
            parser.print_help()
            raise Exception("Error: no input file/directory given")

        # Check if we're doing affordance learning
        if results.objects:
            affordance_plotting(results.objects, results.actions, results.train_types, results.data_file)



if __name__== "__main__":
    main()
    print "done"




