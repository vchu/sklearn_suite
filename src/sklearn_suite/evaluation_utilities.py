#!/usr/bin/env python
# Utility class that deals with testing
#
#
#
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import os
import numpy as np
from collections import defaultdict
from hmm_custom import GaussianHMMClassifier, GMMHMMClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from learning_constants import * # imports all of the constants
from learning_utilities import prepare_data
from data_utilities import load_pkl

def default_to_regular(d):
    '''
    Convenience function that converts defaultdicts
    to regular dictionaries
    '''
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d

def load_hmm_segments(location):
    '''
    Helper function that loads the HMMs and stores them
    in a dictionary to test with
    '''
    seg_clfs = defaultdict(dict)
    for feat_type in os.listdir(location):
        # Go through each of the affordance directories (they have different segment numbers)
        feat_loc = os.path.join(location, feat_type)
        for aff in os.listdir(feat_loc):
            seg_clfs[aff][feat_type] = defaultdict(dict)
            seg_loc = os.path.join(feat_loc, aff)
            for seg in os.listdir(seg_loc):
                loc = os.path.join(seg_loc,seg)
                model_files = [(f,os.path.join(loc,f)) for f in os.listdir(loc)]
                for model in model_files:
                    name = model[0]
                    if not name.endswith('.pkl'):
                        continue
                    else:
                        name = name.split('.pkl')[0].split('_')
                        aff_type = name[-1]
                        seg_clfs[aff][feat_type][int(seg)][aff_type] = load_pkl(model[1])
    seg_clfs = default_to_regular(seg_clfs)
    return seg_clfs

def test_hmm_segment(test_data, classifiers, affordance):
    '''
    Given a set of classifiers by segment, pull out the correct
    segment and return the likelihoods and label

    NOTE: affordance must be of form 'object_action'
    '''

    mode_results = defaultdict(dict) 
    for mode in classifiers[affordance]:
        for seg_num in classifiers[affordance][mode]:
            # Pull out the specific segment
            seg_clf = dict()
            for sf_key in [SUCCESS_KEY,FAIL_KEY]:
                seg_clf[sf_key] = classifiers[affordance][mode][seg_num][sf_key][CLF_KEY]
      
            mode_data = test_data[mode][0][MERGED_FEAT]
            mode_results[mode][seg_num] = defaultdict(dict)
            # For each test run, evaluate for each modality and store
            for i in xrange(len(mode_data)):
                run = mode_data[i]
                data = run[0]
                label = run[1]
                # Pull out the data segment
                state_array = test_data['state'][0][MERGED_FEAT][i][0]
                segments = get_segment(data, seg_num, state_data=state_array, fixed=True)

                # Get the likelihoods for the segment
                likelihoods = dict()
                for seg_tup in [0,1]:
                    for j in xrange(1,len(segments[seg_tup])):
                        for sf_key in [SUCCESS_KEY,FAIL_KEY]:
                            if sf_key not in likelihoods:
                                likelihoods[sf_key] = ([],[])
                            likelihoods[sf_key][seg_tup].append(seg_clf[sf_key].score(segments[seg_tup][0:j]))

                for sf_key in [SUCCESS_KEY,FAIL_KEY]:
                    mode_results[mode][seg_num][i][sf_key] = likelihoods[sf_key]

    truth = [x[1] for x in test_data['state'][0][MERGED_FEAT]]
    return (mode_results, truth)

def plot_segment_results(results):

    import matplotlib.pyplot as plt
    # Massage the results into something easily plottable
    all_aff_results = defaultdict(dict)
    for affordance in results:
        for test_set in results[affordance]:
            feat_store = dict()
            seg_results = results[affordance][test_set]['seg_results']
            truth_vals = seg_results[1]
            segments = seg_results[0]
            for feat_type in segments:
                feat_seg = segments[feat_type]
                feat_store[feat_type] = defaultdict(dict)
                for seg in feat_seg:
                    for run_num in feat_seg[seg]:
                        for sf_key in feat_seg[seg][run_num]:
                            if sf_key not in feat_store[feat_type][run_num]:
                                feat_store[feat_type][run_num][sf_key] = []
                            feat_store[feat_type][run_num][sf_key].append(feat_seg[seg][run_num][sf_key])
            all_aff_results[affordance][test_set] = feat_store

    # now actually plot
    
    import pdb; pdb.set_trace()
    plt.figure(0)
    plt.plot(np.hstack(all_aff_results['lamp_on_users'][0]['force'][0]['fail']), color='r')
    plt.plot(np.hstack(all_aff_results['lamp_on_users'][0]['force'][0]['success']), color='b')

    # Plot the segment locations
    

    plt.figure(1)
    plt.plot(np.hstack(all_aff_results['lamp_on_users'][0]['visual'][0]['fail']), color='r')
    plt.plot(np.hstack(all_aff_results['lamp_on_users'][0]['visual'][0]['success']), color='b')

def get_segment(data, seg_num, state_data= [], fixed=False):
    '''
    Pull out the data depending on if we want the data to be fixed
    or not. Fixed means using the KFs as part of the information
    '''
    # check what kind of segment we need to pull out
    if fixed:
        # compute for the segment afterwards as well
        idx = np.where(state_data == seg_num)[0]
        idx2 = np.where(state_data == seg_num+1)[0]
        return (data[idx,:],data[idx2,:])
    else:
        # Keep track of what segments were considered already part of a previous segment
        import pdb; pdb.set_trace()
        return None

def test_hmm(test_data, classifier_dict, thresh=False, score_report=True):

    # Pull out the data
    test_X, test_Y = prepare_data(test_data) 
 
    # If key exists that means we have TWO HMMs 
    if SUCCESS_KEY in classifier_dict:
        pos_hmm = classifier_dict[SUCCESS_KEY]
        neg_hmm = classifier_dict[FAIL_KEY] 
    else:
        pos_hmm = classifier_dict
        neg_hmm = None

    if thresh:
        # Compute threshold
        (threshold, s_mean, f_mean) = get_mean_likelihood(pos_hmm[CLF_KEY], classifier_dict) 
 
    # Pull out classifier
    clf = pos_hmm[CLF_KEY]

    # If we have two models
    if neg_hmm is not None:
        clf_neg = neg_hmm[CLF_KEY]

    #import pdb; pdb.set_trace()
    # Store prediction vector
    predict = []

    for feat_vec in test_X:

        success_score = clf.score([feat_vec])

        # If we want to compute using a threshold rather than the 
        # negative model
        if thresh:
            if success_score > threshold:
                predict.append(SUCCESS_VAL)
            else:
                predict.append(FAIL_VAL)

        # If we have two models
        elif neg_hmm is not None:

            fail_score = clf_neg.score([feat_vec])
            if success_score > fail_score:
                predict.append(SUCCESS_VAL)
            else:
                predict.append(FAIL_VAL)

        #else: # Do something with hard coded value?!

    # Print results
    if score_report:
        print (classification_report(test_Y,predict))

    return (predict, test_Y)


def get_mean_likelihood(hmm, classifier_dict):

    success_features = classifier_dict[SUCCESS_KEY][TRAIN_KEY][FEAT_KEY] 
    success_scores = []
    for value in success_features:
        success_scores.append(hmm.score([value]))

    fail_features = classifier_dict[FAIL_KEY][TRAIN_KEY][FEAT_KEY] 
    fail_scores = []
    for value in fail_features:
        fail_scores.append(hmm.score([value]))

    threshold = np.mean((np.mean(success_scores),np.mean(fail_scores)))
    success_mean = np.mean(success_scores)
    fail_mean = np.mean(fail_scores)

    print "success likelihood mean: %s" % str(np.mean(success_scores))
    print "fail likelihood mean: %s"  % str(np.mean(fail_scores))
    print "Threshold is: %d" % threshold 

    return (threshold, success_mean, fail_mean)



def compute_results(results, pos_label=0):

    # Go through each skill and print them in latex table format
    for filename in results:

        # Pull out the data for that file 
        fileset = results[filename]
        fileset_names = fileset.keys()
        fileset_names.sort()

        precision_arr = []
        recall_arr = []
        f1_arr = []
        acc_arr = []
        print '###############################################'
        print filename
        print '###############################################'
        
        for testset in fileset_names:

            # Pull out predicted_Y
            predict_Y = fileset[testset][PREDICTY_KEY] 

            # Pull out test_Y
            test_Y = fileset[testset][TESTY_KEY] 

            print '\npredict: '+str(predict_Y)
            print 'actual: %s\n' % str(test_Y)

            # Compute F1, precision, recall 
            # if we want weighted...
            #precision = precision_score(test_Y, predict_Y, pos_label=None, average='weighted')
            #recall = recall_score(test_Y, predict_Y, pos_label=None, average='weighted')
            #f1 = f1_score(test_Y, predict_Y, pos_label=None, average='weighted')
            precision = precision_score(test_Y, predict_Y, pos_label=pos_label)
            recall = recall_score(test_Y, predict_Y, pos_label=pos_label)
            f1 = f1_score(test_Y, predict_Y, pos_label=pos_label)
            accuracy = accuracy_score(test_Y, predict_Y)

            print classification_report(test_Y, predict_Y)
            print 'accuracy: %f\n' % accuracy
    
            # Store away
            results[filename][testset][PRECISION] = precision
            results[filename][testset][RECALL] = recall
            results[filename][testset][F1] = f1
            results[filename][testset][ACCURACY] = accuracy
            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1)
            acc_arr.append(accuracy)

        # Store away the average of the test sets
        results[filename][M_PRECISION] = np.mean(precision_arr)
        results[filename][M_RECALL] = np.mean(recall_arr)
        results[filename][M_F1] = np.mean(f1_arr)
        results[filename][M_ACCURACY] = np.mean(acc_arr)

    return results

def print_avg_results_latex(results):
    '''
    Generic printing function that does not order or take into account 
    the scores other than the filename
    '''

    # Go through each skill and print them in latex table format
    filenames = results.keys()
    filenames.sort()
    for filename in filenames:

        # Pull out the data for that file 
        fileset = results[filename]

        all_score = []
        all_score.append(fileset[M_PRECISION])
        all_score.append(fileset[M_RECALL])
        all_score.append(fileset[M_F1])

        # Create the actual latex string
        print_string = print_latex_string([filename.replace('_',' ').title()], all_score)
        print print_string

def print_avg_results_latex_new(results):
    '''
    Generic printing function that does not order or take into account 
    the scores other than the filename
    '''

    # Go through each skill and print them in latex table format
    filenames = results.keys()
    filenames.sort()
    for filename in filenames:

        # Pull out the data for that file 
        fileset = results[filename]

        all_score = []
        all_score.append(fileset[M_PRECISION])
        all_score.append(fileset[M_RECALL])
        all_score.append(fileset[M_F1])
        all_score.append(fileset[M_ACCURACY])

        # Create the actual latex string
        print_string = print_latex_string([filename.replace('_',' ').title()], all_score)
        print print_string

def print_avg_affordance_guided_latex(results):
    '''
    prints the latex table that includes the per user breakdown of results
    '''
    affordance_results = dict()

    # Go through each skill and print them in latex table format
    for filename in results:

        # Pull out the data for that file 
        fileset = results[filename]

        # store each affordance properly
        aff_name = filename.split("_")  
        obj = aff_name[0]
        action = aff_name[1]
        train_type = '_'.join(aff_name[2:4])
        user_type = aff_name[-1]

        all_score = []
        all_score.append(fileset[M_PRECISION])
        all_score.append(fileset[M_RECALL])
        all_score.append(fileset[M_F1])

        # Check if this is the first time we encountered this object
        if obj not in affordance_results:
            affordance_results[obj] = dict()

        # Check if this is the first time we encountered this action
        if action not in affordance_results[obj]:
            affordance_results[obj][action] = defaultdict(dict)

        # Store away the results
        affordance_results[obj][action][train_type][user_type] = all_score
    
    objects = ['breadbox','pastajar','drawer']
    all_user_IDs = ['subject3','subject4','subject5','subject6','subject7','subject8','subject9', 'subject10', 'subject11', 'subject12']
    # Now go through the affordance dictionary and print just the rows we want
    #for obj in affordance_results: # random order
    for obj in objects: # enforce print object order
        for act in affordance_results[obj]:

            print "\multirow{5}{*}{"+ obj.title() + " " + act.title() + "}"

            print_scores_user = []
            print_scores_strat4 = []
            print_scores_strat5 = []
            for userID in all_user_IDs:

                # Check if we have scores this for user
                if userID in affordance_results[obj][act]['user_study']:
                    print_scores_user.append(affordance_results[obj][act]['user_study'][userID])

                if userID in affordance_results[obj][act]['strategy_4']:
                    print_scores_strat4.append(affordance_results[obj][act]['strategy_4'][userID])

                if userID in affordance_results[obj][act]['strategy_5']:
                    print_scores_strat5.append(affordance_results[obj][act]['strategy_5'][userID])

            # Pull out self exploraiton scores
            self_scores = affordance_results[obj][act]['self_exploration']['users']
            self_std_scores = ['skip','skip','skip']

            # Pull out guided aggreg bias scores 
            if 'guided_exploration' not in affordance_results[obj][act]:
                guided_scores = ['N/A','N/A','N/A']
            else:
                guided_scores = affordance_results[obj][act]['guided_exploration']['users'] 
            guided_std_scores = ['skip','skip','skip']


            # Scores for per user study scores
            print_scores_user = np.vstack(print_scores_user)
            print_scores_strat4 = np.vstack(print_scores_strat4)
            print_scores_strat5 = np.vstack(print_scores_strat5)

            # compute mean and std for each metric (P,R,F_1)
            user_avg_scores = np.mean(print_scores_user, axis=0).tolist()
            strat4_avg_scores = np.mean(print_scores_strat4, axis=0).tolist()
            strat5_avg_scores = np.mean(print_scores_strat5, axis=0).tolist()


            all_scores = self_scores+user_avg_scores+guided_scores+strat4_avg_scores+strat5_avg_scores

            # Compute the std for each metric
            user_std_scores = np.std(print_scores_user, axis=0).tolist()
            strat4_std_scores = np.std(print_scores_strat4, axis=0).tolist()
            strat5_std_scores = np.std(print_scores_strat5, axis=0).tolist()
            all_std = self_std_scores+user_std_scores+guided_std_scores+strat4_std_scores+strat5_std_scores

            #print_string = print_latex_string([obj,act],all_scores, all_std)

            # print each of them separately
            print '& '+print_latex_string(['Self'],self_scores)
            print '& '+print_latex_string_std(['''Supervised\\tnote{a}'''],user_avg_scores, user_std_scores)
            print '& '+print_latex_string(['Aggregate'],guided_scores)
            print '& '+print_latex_string_std(['''Iconic\\tnote{a}'''], strat4_avg_scores, strat4_std_scores)
            print '& '+print_latex_string_std(['''Boundary\\tnote{a}'''], strat5_avg_scores, strat5_std_scores)
            print '\hline'


def print_affordance_user_latex(results):
    '''
    prints the latex table that includes the per user breakdown of results
    '''
    affordance_results = dict()

    # Go through each skill and print them in latex table format
    for filename in results:

        # Pull out the data for that file 
        fileset = results[filename]

        # store each affordance properly
        aff_name = filename.split("_")  
        obj = aff_name[0]
        action = aff_name[1]
        train_type = '_'.join(aff_name[2:4])
        user_type = aff_name[-1]

        all_score = []
        all_score.append(fileset[M_PRECISION])
        all_score.append(fileset[M_RECALL])
        all_score.append(fileset[M_F1])

        # Check if this is the first time we encountered this object
        if obj not in affordance_results:
            affordance_results[obj] = dict()

        # Check if this is the first time we encountered this action
        if action not in affordance_results[obj]:
            affordance_results[obj][action] = defaultdict(dict)

        # Store away the results
        affordance_results[obj][action][train_type][user_type] = all_score
    
    objects = ['breadbox','pastajar','drawer']
    all_user_IDs = ['subject3','subject4','subject5','subject6','subject7','subject8','subject9', 'subject10', 'subject11', 'subject12']
    # Now go through the affordance dictionary and print just the rows we want
    #for obj in affordance_results: # random order
    for obj in objects: # enforce print object order
        for act in affordance_results[obj]:
            print "\multirow{3}{*}{"+ obj.title() + " " + act.title() + "}"
            for userID in all_user_IDs:

                # Check if we have scores this for user
                if userID in affordance_results[obj][act]['user_study']:
                    print_scores_user = affordance_results[obj][act]['user_study'][userID]
                else:
                    print_scores_user = ['N/A', 'N/A', 'N/A']

                if userID in affordance_results[obj][act]['strategy_4']:
                    print_scores_strat4 = affordance_results[obj][act]['strategy_4'][userID]
                else:
                    print_scores_strat4 = ['N/A', 'N/A', 'N/A']

                if userID in affordance_results[obj][act]['strategy_5']:
                    print_scores_strat5 = affordance_results[obj][act]['strategy_5'][userID]
                else:
                    print_scores_strat5 = ['N/A', 'N/A', 'N/A']

                all_scores = print_scores_user+print_scores_strat4+print_scores_strat5
                userID = '& User '+str(int(userID.split('t')[-1])-2)
                print_string = print_latex_string([userID],all_scores)
                if all_scores.count('N/A') < 9:
                    print print_string
            print '\hline'

def print_affordance_latex(results):

    affordance_results = dict()
    # Go through each skill and print them in latex table format
    for filename in results:

        # Pull out the data for that file 
        fileset = results[filename]

        # store each affordance properly
        aff_name = filename.split("_")  
        obj = aff_name[0]
        action = aff_name[1]
        train_type = '_'.join(aff_name[2:4])

        all_score = []
        all_score.append(fileset[M_PRECISION])
        all_score.append(fileset[M_RECALL])
        all_score.append(fileset[M_F1])

        # Check if this is the first time we encountered this object
        if obj not in affordance_results:
            affordance_results[obj] = defaultdict(dict)

        affordance_results[obj][action][train_type] = all_score

    objects = ['breadbox','pastajar','drawer']
    # Now go through the affordance dictionary and print just the rows we want
    #for obj in affordance_results: # random order
    for obj in objects: # enforce print object order
        for act in affordance_results[obj]:

            self_scores = affordance_results[obj][act]['self_exploration'] 
            user_scores = affordance_results[obj][act]['user_study'] 
            if 'guided_exploration' not in affordance_results[obj][act]:
                guided_scores = ['N/A','N/A','N/A']
            else:
                guided_scores = affordance_results[obj][act]['guided_exploration'] 

            print_string = print_latex_string([obj,act],self_scores+user_scores+guided_scores)
            print print_string

def print_latex_string(prefix, scores):
    '''
    Takes in a list of scores and prints them out in latex form
    
    Also takes in a list of prefix strings to also add
    '''

    # Create the initial string and add all of the prefix values
    print_string = ""
    for pre in prefix:
        print_string = print_string + pre.title() + ' & '

    # Attaches each of the scores in whatever order they came in
    for score in scores:
        if isinstance(score, float):
            score_string = "{:.2f}".format(score)
            score_string_color = "{:.2f}".format(score*100)
        else:
            score_string = 'N/A'
            score_string_color = '0' 

        print_string = print_string + ' \cellcolor{gray!'+score_string_color+'} '+score_string +'  &'

    print_string = print_string[0:-1] + "\\\\"
    return print_string

def print_latex_string_std(prefix, mean_scores, std_values):
    '''
    Takes in a list of scores and prints them out in latex form
    
    Also takes in a list of prefix strings to also add
    '''

    # Create the initial string and add all of the prefix values
    print_string = ""
    for pre in prefix:
        print_string = print_string + pre.title() + ' & '

    # Attaches each of the scores in whatever order they came in
    for i in xrange(len(mean_scores)):
        mean_val = mean_scores[i]
        var_val = std_values[i]

        if isinstance(mean_val, float):
            score_string = "{:.2f}".format(mean_val)
            score_string_color = "{:.2f}".format(mean_val*100)
        else:
            score_string = 'N/A'
            score_string_color = '0' 

        # Check if we have std values to print out
        if var_val == 'skip':
            print_string = print_string + ' \cellcolor{gray!'+score_string_color+'} '+score_string +'  &'

        else:
            std_string = "{:.2f}".format(var_val)

            print_string = print_string + ' \cellcolor{gray!'+score_string_color+'} '+score_string +'$\pm$'+std_string+'  &'

    print_string = print_string[0:-1] + "\\\\"
    return print_string






