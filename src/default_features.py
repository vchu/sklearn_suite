#!/usr/bin/env python
# Default feature class
#
# Currently contains the features used for haptic affordances
# and social affordances exploration
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy

from data_utilities import check_data # basic function that cleans data

# Constants for feature extraction
FT_NORM_FLAG = True
DEFAULT_STATE_NAME = 'C6_FSM_state'


def compute_features(all_data, skip_topics=['skill'], state_name=DEFAULT_STATE_NAME, ft_norm=FT_NORM_FLAG):
    '''
    Simple function that takes the data in the form of a python dictionary and puts it in a form
    that is easy to extend to features

    Input: loaded raw data using load_data or load_pkl
    Output: python dictionary of features
    '''

    import pdb; pdb.set_trace()
    feature_store = defaultdict(dict)

    # Set type are "fail" or "success"
    for set_type in all_data:

        data = all_data[set_type]
        if set_type in skip_topics:
            continue

        # Concat the features into one array
        for run_name in data:
            run_data = data[run_name]

            # Store off data
            run_dict = dict()
            run_dict['state'] = check_data(run_data[state_name]['data'])
            if 'KF_current_frame' in run_data:
                run_dict['KF'] = check_data(run_data['KF_current_frame']['data'])
            else:
                run_dict['KF'] = None

            # Pull out information about objects
            objects = run_data['c6_tracked_objects_clusters_transformed']
            for object_num in objects:

                # Create entry for each object
                object_data = objects[object_num]
                run_dict[object_num] = defaultdict(dict)

                for viz_feat in object_data:

                    feat_data = object_data[viz_feat]
                    if viz_feat == 'compactness':
                        feat_data = feat_data/COMPACT_NORM

                    if viz_feat == 'rgba_color':
                        feat_data = feat_data/RGBA_COLOR_NORM
                        feat_data = feat_data[:,0:3]

                    run_dict[object_num][viz_feat] = check_data(feat_data)

            for arm in arms:

                force = check_data(run_data['loadx6_'+arm+'_state']['force'])
                torque = check_data(run_data['loadx6_'+arm+'_state']['torque'])

                # normalize the f/t
                if ft_norm:
                    run_dict[arm+'_force'] = np.mean(force[0:10], axis=0)-force
                    run_dict[arm+'_torque'] = np.mean(torque[0:10], axis=0)-torque
                else:
                    run_dict[arm+'_force'] = force
                    run_dict[arm+'_torque'] = torque

                # Get all the arm joint names
                arm_joints = [j for j in run_data['humanoid_state']['name'] if arm+'_arm' in j]
                arm_idx = np.in1d(run_data['humanoid_state']['name'], np.array(arm_joints))

                # Store off the arm features
                run_dict[arm+'_joint_position'] =  check_data(run_data['humanoid_state']['position'][:,arm_idx])
                run_dict[arm+'_joint_effort'] =  check_data(run_data['humanoid_state']['effort'][:,arm_idx])
                run_dict[arm+'_joint_velocity'] =  check_data(run_data['humanoid_state']['velocity'][:,arm_idx])
                run_dict[arm+'_joint_name'] =  check_data(run_data['humanoid_state']['name'][:,arm_idx])

                # Create a set of "future" positions to train on
                # Pop off the front to shift and then append 0 to the end?
                run_dict[arm+'_joint_position_future'] = np.insert(np.delete(run_dict[arm+'_joint_position'],0,axis=0),-1,0,axis=0)

            feature_store[set_type][run_name] = run_dict

    # Retain some of the extra names we want to keep (filename, etc)
    for extra_store_names in skip_topics:
        feature_store[extra_store_names] = all_data[extra_store_names]

    return feature_store



