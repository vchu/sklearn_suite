#!/usr/bin/env python
# Default feature class
#
# Currently contains the features used for haptic affordances
# and social affordances exploration
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import numpy as np
from scipy import signal
from collections import defaultdict
from data_utilities import check_data # basic function that cleans data
from learning_constants import * # All of the constants defined
from tf import transformations as transform
from pydub import AudioSegment
import python_speech_features as psf
import cv2

# Constants for feature extraction
FT_NORM_FLAG = True
DEFAULT_STATE_NAME = 'C6_FSM_state'
KF_STATE_NAME = 'KF_current_frame'
OBJECT_TRACKER_TOPIC = 'c6_tracked_objects_clusters_transformed'
HLPR_OBJECT_TRACKER = '_beliefs_labels'
IMAGE_TOPIC = '_kinect_qhd_image_color_rect_compressed'
COMPACT_FEAT = 'compactness'
RGBA_FEAT = 'rgba_color'
arms = ['right','left']

def compute_features(all_data, state_name=DEFAULT_STATE_NAME, ft_norm=FT_NORM_FLAG, jaco_flag=False, tracked_object=None):
    '''
    Simple function that takes the data in the form of a python dictionary and puts it in a form
    that is easy to extend to features

    Input: loaded raw data using load_data or load_pkl
    Output: python dictionary of features
    '''

    feature_store = dict()
    feature_store[FEAT_DICT_KEY] = defaultdict(dict)
    data_only = all_data[DATA_KEY]

    # Set type are "fail" or "success"
    for set_type in data_only:

        data = data_only[set_type]

        # Concat the features into one array
        for run_name in data:
            run_data = data[run_name]

            # Store off data
            run_dict = dict()
            run_dict['state'] = check_data(run_data[state_name][DATA_KEY])
            run_dict['time'] = check_data(run_data[state_name]['time'])
            if KF_STATE_NAME in run_data:
                run_dict['KF'] = check_data(run_data[KF_STATE_NAME][DATA_KEY])
            else:
                run_dict['KF'] = None

            ########################################################
            # Visual Features
            ########################################################
            # Pull out information about objects
            if jaco_flag:
                parse_obj = False
                if 'objects' in run_data[HLPR_OBJECT_TRACKER].keys():
                    objects = run_data[HLPR_OBJECT_TRACKER]['objects']
                    if 'labels' in run_data[HLPR_OBJECT_TRACKER].keys():
                        labels = run_data[HLPR_OBJECT_TRACKER]['labels']
                        if 'labels0' in labels:
                            parse_obj = True

                # Pull out the object we're trying to track
                if tracked_object and parse_obj:
                    # Pull out the first label and see if we can check the length
                    lab_obj = labels['labels0']
                    if len(lab_obj['data']) != len(run_data[state_name][DATA_KEY]):
                        rospy.logwarn("Object label lengths don't match!")
                   
                    # Assume only the first object label will have the objects we need 
                    idx = np.where(lab_obj['data'] == tracked_object)[0]

                    # Cycle through the features we want
                    VIZ_FEATURES = ['bb_translation','bb_rotation','rgba_color','bb_volume','bb_dimensions','obj_squareness']
                    object_name = 'objects0'
                    run_dict[object_name] = dict()
                    # Position of object
                    object_trans = objects[object_name]['transform']['translation']
                    # Orientation of object
                    object_rot = objects[object_name]['transform']['rotation']
                    # Object color
                    object_color = objects[object_name]['basicInfo']['rgba_color']
                    # Dimension of BB
                    object_dim = objects[object_name]['obb']['bb_dims']
                    # Pointcloud info
                    object_num_points = objects[object_name]['basicInfo']['num_points']

                    # Now actually store away the visual features
                    last_obj_dict = dict()
                    store_dict = dict()
                    for viz_feature in VIZ_FEATURES:
                        store_dict[viz_feature] = []

                    for i in xrange(len(object_trans['x'])):
                        if i in idx:
                            store_dict['bb_translation'].append([object_trans['x'][i],object_trans['y'][i],object_trans['z'][i]])
                            store_dict['bb_rotation'].append([object_rot['x'][i],object_rot['y'][i],object_rot['z'][i],object_rot['w'][i]])
                            store_dict['rgba_color'].append([object_color['r'][i],object_color['g'][i],object_color['b'][i],object_color['a'][i]])
                            store_dict['bb_volume'].append(object_dim['x'][i]*object_dim['y'][i]*object_dim['z'][i])
                            store_dict['bb_dimensions'].append([object_dim['x'][i],object_dim['y'][i],object_dim['z'][i]])
                            store_dict['obj_squareness'].append(object_num_points[i]/store_dict['bb_volume'][i])
                            for viz_feature in VIZ_FEATURES:
                                last_obj_dict[viz_feature] = store_dict[viz_feature][-1]
                        else:
                            for viz_feature in VIZ_FEATURES:
                                store_dict[viz_feature].append(last_obj_dict[viz_feature])
                    for viz_feature in VIZ_FEATURES:
                        run_dict[viz_feature] = np.vstack(store_dict[viz_feature])

                else:
                    rospy.logwarn("No given object, skipping")
                
            else:
                objects = run_data[OBJECT_TRACKER_TOPIC]
                for object_num in objects:

                    # Create entry for each object
                    object_data = objects[object_num]
                    run_dict[object_num] = defaultdict(dict)

                    for viz_feat in object_data:

                        feat_data = check_data(object_data[viz_feat])
                        if viz_feat == COMPACT_FEAT:
                            feat_data = feat_data/COMPACT_NORM

                        if viz_feat == RGBA_FEAT:
                            feat_data = feat_data/RGBA_COLOR_NORM
                            feat_data = feat_data[:,0:3]

                        run_dict[object_num][viz_feat] = feat_data

            if jaco_flag:
                arms = ['jaco']

            for arm in arms:
                ########################################################
                # Force/Torque features
                ########################################################

                if jaco_flag:
                    data_store = dict()
                    wrench_run = run_data['_j2s7s300_driver_out_tool_wrench']['wrench']
                    for ft in ['force','torque']:
                        data_store[ft] = []
                        for channel in ['x','y','z']:
                            data_store[ft].append(wrench_run[ft][channel])
                    force = np.column_stack(data_store['force'])
                    torque = np.column_stack(data_store['torque'])
                else:
                    force = check_data(run_data['loadx6_'+arm+'_state']['force'])
                    torque = check_data(run_data['loadx6_'+arm+'_state']['torque'])

                # normalize the f/t
                if ft_norm:
                    run_dict[arm+'_force'] = np.mean(force[0:10], axis=0)-force
                    run_dict[arm+'_torque'] = np.mean(torque[0:10], axis=0)-torque
                else:
                    run_dict[arm+'_force'] = force
                    run_dict[arm+'_torque'] = torque

                ########################################################
                # Joint level (per joint position/orientation/etc.)
                ########################################################

                if jaco_flag:
                    arm_topic = '_j2s7s300_driver_out_joint_state'
                    arm_idx = [0,1,2,3,4,5,6]
                else:
                    arm_topic = 'humanoid_state'
                    # Get all the arm joint names
                    arm_joints = [j for j in run_data['humanoid_state']['name'] if arm+'_arm' in j]
                    arm_idx = np.in1d(run_data['humanoid_state']['name'], np.array(arm_joints))

                # Store off the arm features
                run_dict[arm+'_joint_position'] =  check_data(run_data[arm_topic]['position'][:,arm_idx])
                run_dict[arm+'_joint_effort'] =  check_data(run_data[arm_topic]['effort'][:,arm_idx])
                run_dict[arm+'_joint_velocity'] =  check_data(run_data[arm_topic]['velocity'][:,arm_idx])
                run_dict[arm+'_joint_name'] =  check_data(run_data[arm_topic]['name'][arm_idx])

                # Create a set of "future" positions to train on
                # Pop off the front to shift and then append 0 to the end?
                run_dict[arm+'_joint_position_future'] = np.insert(np.delete(run_dict[arm+'_joint_position'],0,axis=0),-1,0,axis=0)

                ########################################################
                # Hand Joint level (per joint position/orientation/etc.)
                ########################################################

                if jaco_flag:
                    gripper_topic = '_vector_right_gripper_stat'
                    run_dict[arm+'_hand_joint_position'] = run_data[gripper_topic]['position']
                    run_dict[arm+'_hand_joint_req_pos'] = run_data[gripper_topic]['requested_position']
                    run_dict[arm+'_hand_joint_current'] = run_data[gripper_topic]['current']
                else:
                    # Get features related to the hand
                    hand_joints = [j for j in run_data['humanoid_state']['name'] if arm+'_hand' in j]
                    hand_idx = np.in1d(run_data['humanoid_state']['name'], np.array(hand_joints))

                    # Store off the hand features
                    run_dict[arm+'_hand_joint_position'] =  check_data(run_data['humanoid_state']['position'][:,hand_idx])
                    run_dict[arm+'_hand_joint_effort'] =  check_data(run_data['humanoid_state']['effort'][:,hand_idx])
                    run_dict[arm+'_hand_joint_velocity'] =  check_data(run_data['humanoid_state']['velocity'][:,hand_idx])
                    run_dict[arm+'_hand_joint_name'] =  check_data(run_data['humanoid_state']['name'][hand_idx])

                ########################################################
                # End Effector (EEF)  Features
                ########################################################

                if jaco_flag:
                    pose_topic = '_eef_pose'
                else:
                    pose_topic = 'c6_end_effect_pose_'+arm

                # Get EEF information
                # Note: orientation is a quaternion in the form [qx,qy,qz,qw]
                run_dict[arm+'_EEF_position'] = check_data(run_data[pose_topic]['position'])
                run_dict[arm+'_EEF_orientation'] = check_data(run_data[pose_topic]['orientation'])

                # Compute EEF position relative to object (only for first object)
                if jaco_flag:
                    if parse_obj:
                        # Position - vector from Object to EEF
                        run_dict[arm+'_EEF_obj'] = run_dict[arm+'_EEF_position'] - run_dict['bb_translation']
                        qz_object = run_dict['bb_rotation']
                        qz_object_inv = map(transform.quaternion_inverse, qz_object)
                        Rot_obj_2_EEF = map(lambda x,y: transform.quaternion_multiply(x,y),run_dict[arm+'_EEF_orientation'], qz_object_inv)
                        run_dict[arm+'_EEF_obj_quat'] = Rot_obj_2_EEF
                else:
                    objects = run_data[OBJECT_TRACKER_TOPIC]
                    for object_num in objects:
                        # Position - vector from Object to EEF
                        run_dict[arm+'_EEF_obj'] = run_dict[arm+'_EEF_position'] - run_dict[object_num]['centroid']
                    
                        # Orientation 
                        # Create quaternion from object angle (about zaxis (0,0,1))
                        # Rot_obj_2_EEF = q_eff * inv(q_obj)
                        qz_object = map(lambda x: transform.quaternion_about_axis(x,(0,0,1)), run_dict[object_num]['angle'])
                        qz_object_inv = map(transform.quaternion_inverse, qz_object)
                        Rot_obj_2_EEF = map(lambda x,y: transform.quaternion_multiply(x,y),run_dict[arm+'_EEF_orientation'], qz_object_inv)
                        run_dict[arm+'_EEF_obj_quat'] = Rot_obj_2_EEF

                if jaco_flag:
                    ########################################################
                    # Audio Features
                    ########################################################
                    segment = np.hstack(run_data['_kinect_audio']['raw_audio'])
                    # Write data to mp3 file
                    temp_file_name = 'temp_audio.mp3'
                    mp3_file = open(temp_file_name,'w')
                    mp3_file.write(segment.tobytes())
                    mp3_file.close()
                    data_stream = AudioSegment.from_mp3(temp_file_name)
                    audio_data = data_stream._data
                    converted_audio = np.fromstring(audio_data, dtype=np.int16)
                    run_dict['audio_data'] = converted_audio
                    run_dict['audio_mfcc'] = psf.mfcc(converted_audio)

                    ########################################################
                    # Image
                    ########################################################
                    # Convert the image and store as a opencv image
                    img_data_arr = run_data[IMAGE_TOPIC]['data']
                    convert_arr = []
                    for img in img_data_arr:
                        encode = cv2.imdecode(img, cv2.CV_LOAD_IMAGE_COLOR)
                        convert_arr.append(encode)
                   
                    run_dict['image_data'] = convert_arr 
            
            feature_store[FEAT_DICT_KEY][set_type][run_name] = run_dict

    # Keep original data with the features 
    for extra_store_names in all_data:
        feature_store[extra_store_names] = all_data[extra_store_names]

    return feature_store


def extract_features_specific_keys(all_data, run_keys, feature_list, object_num_array=[], object_feat_list=[]):
    '''
    This function takes a python dictionary of features, features to extract
    and converts them into a list of triples for specific keys
    '''

    # Create data storage to use to call extract_features
    selected_data = defaultdict(dict) 
    selected_data[FEAT_DICT_KEY][SUCCESS_KEY] = dict()
    selected_data[FEAT_DICT_KEY][FAIL_KEY] = dict()

    # Cycle through train/test sets
    for tt_set in all_data:

        # Pull out the feature dictionary
        t_data = all_data[tt_set][FEAT_DICT_KEY]

        for sf_type in t_data:
            run_data = t_data[sf_type]

            # Cycle through run_data and save
            for run_name in run_data:
                if run_name in run_keys:
                    selected_data[FEAT_DICT_KEY][sf_type][run_name] = run_data[run_name]

    # Sanity check the data - did we load all of the keys
    if len(selected_data[FEAT_DICT_KEY][SUCCESS_KEY]) + len(selected_data[FEAT_DICT_KEY][FAIL_KEY]) != len(run_keys):
        print 'WARNING: All keys not loaded. Stopping loading.'
        import pdb; pdb.set_trace()

    # Pull out the data using the existing method
    return extract_features(selected_data, feature_list, object_num_array=object_num_array, object_feat_list=object_feat_list)


def extract_features(all_data, feature_list, object_num_array=[], object_feat_list=[]):
    '''
    This function takes a python dictionary of features, features to extract
    and converts them into a list of triples
    '''

    only_feat = all_data[FEAT_DICT_KEY] # pull out the features from the dictionary
    norm_data = dict()
    keys = dict()

    merged_data = []
    merged_keys = dict()
    merged_keys[NAME_KEY] = []
    merged_keys[RUN_KEY] = []
    merged_keys[LABEL_KEY] = []

    # Set type are "fail" or "success"
    for set_type in only_feat:

        data = only_feat[set_type]

        norm_data[set_type] = []
        keys[set_type] = dict()
        keys[set_type][NAME_KEY] = []
        keys[set_type][RUN_KEY] = []
        keys[set_type][LABEL_KEY] = []

        # Concat the forces and torques into one array
        # Sort the feature list
        all_run_names = data.keys()
        all_run_names.sort()
        for run_name in all_run_names:
            one_run = data[run_name]

            feature_to_stack = []
            # Go through all of the features and store in array
            for feature in feature_list:
                feature_to_stack.append(one_run[feature])

            for object_num in object_num_array:
                object_name = OBJECT_CLUSTER_FEAT_PREFIX+str(object_num)

                for object_feat in object_feat_list:
                    feature_to_stack.append(one_run[object_name][object_feat])

            # column stack the features into one matrix
            features = np.column_stack(feature_to_stack)

            # Store the label of the data - if unsupervised - all will be success            
            if set_type == SUCCESS_KEY:
                norm_data[set_type].append((features, SUCCESS_VAL))
                merged_data.append((features, SUCCESS_VAL))
            elif set_type == FAIL_KEY:
                norm_data[set_type].append((features, FAIL_VAL))
                merged_data.append((features, FAIL_VAL))
            else:
                Print("Warning: feature type unknown")

            # Optionally store information about KF for training
            if (data[run_name]['KF'] == None):
                n_states = np.nan
            else:
                n_states = len(np.unique(data[run_name]['KF']))
            #norm_data['n_states'] = n_states

            keys[set_type][NAME_KEY].append(run_name)
            keys[set_type][RUN_KEY].append(int(run_name.split('_')[-1]))
            merged_keys[NAME_KEY].append(run_name)
            merged_keys[RUN_KEY].append(int(run_name.split('_')[-1]))
 
    all_data[FEAT_KEY] = norm_data
    all_data[MERGED_FEAT] = merged_data
    all_data[MERGED_FEAT_KEYS] = merged_keys
    return (all_data, keys)

         
def get_features(data, features, object_features, state_name=None, ft_norm=False, object_num_array=[0]):
    '''
    This is the general API to features 

    Note: ALL feature extraction functions must adhere to the output of this function
    '''
   
    # Check if we have a custom state name
    if state_name is None:
        all_features = compute_features(data, ft_norm=ft_norm)
    else:
        all_features = compute_features(data, state_name=state_name, ft_norm=ft_norm)

    # Extract the features to train on
    (dataset, keys) = extract_features(all_features, features, object_num_array = object_num_array, object_feat_list=object_features)

    return (dataset, keys)

def extract_features_simple(all_features, features, object_num_array=[0], object_feature_list=[]):
    # Extract the features to train on
    (dataset, keys) = extract_features(all_features, features, object_num_array = object_num_array, object_feat_list=object_feature_list)

    return (dataset['features'],keys)



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'flattop']:
    #    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('signal.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
