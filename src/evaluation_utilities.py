#!/usr/bin/env python
# Utility class that deals with testing
#
#
#
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import numpy as np
from collections import defaultdict
from hmm_custom import GaussianHMMClassifier, GMMHMMClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from learning_constants import * # imports all of the constants
from learning_utilities import prepare_data

def test_hmm(test_data, classifier_dict, thresh=False):

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






