#!/usr/bin/env python
# Main data utility class that deals with ML algorithm wrappers
#
# This assumes that data was loaded into numpy arrays where
# the format is nxm (n = examples, m= features)
#
#
#

import roslib; roslib.load_manifest("sklearn_suite")
import rospy
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import LeaveOneOut, LeavePOut
from hmm_custom import GaussianHMMClassifier, GMMHMMClassifier
from sklearn.grid_search import GridSearchCV
from learning_constants import * # imports all of the constants

# Currently put some default values here.. might move it later
# For Cross Validation
DEFAULT_N_FOLD = 3
DEFAULT_P = 2
DEFAULT_JOBS = 1

# For HMMs
DEFAULT_N_COMPONENTS = range(2,7)
DEFAULT_VERBOSE = 5 # 5 is maximum
DEFAULT_ITER = [1000]
DEFAULT_COVAR = ['diag','full'] # options: ['diag','tied','spherical','full']

# For SVMs
DEFAULT_SVM_C = np.linspace(1,1e6,100)
DEFAULT_SVM_KERNELS = ['linear','poly','rbf','sigmoid']
DEFAULT_SVM_PENALTY = ['l1','l2']
DEFAULT_SVM_WEIGHT = None # option: 'auto' to not have equal weight
DEFAULT_SVM_PROB = True
DEFAULT_SVM_LOSS = ['hinge'] # option: 'square_hinge'
DEFAULT_SVM_DUAL = False # perform in the dual space
 
def _cv_setup_helper(cv, num_train, indices=None):
    '''
    Helper to training grid search methods that actually setup
    the cv used

    Input: cv - type of cross validation used
                 (1) None - no cross validation
                 (2) loocv - leave one out cross validation
                 (3) lPOut - leave "P" out cross validation (need to specify p)
                 (4) integer value - ex. 5 (5 fold cross-validation)
                 (5) default is n-fold default cross validation
    '''
    
    print 'train size is: %d' % num_train

    if cv == None: # Rare to not use
        cv = None

    # Check if we're doing loocv or lpout cv
    elif cv == 'loocv':
        print 'Performing LOOCV'
        cv = sklearn.cross_validation.LeavePOut(n=num_train, p=1, indices=indices);

    elif cv == 'lPOut':
        print 'Performing LPOutCV: %d' % p
        cv = sklearn.cross_validation.LeavePOut(n=num_train, p=p, indices=indices);

    # Check if passed in an number to do n-fold CV
    elif isinstance(cv, int):
        cv = min(cv,num_train) #verify size is within data size
        if (cv < 2):
            cv = None

        print 'Performing %d-Fold CV' % cv

    else:
        print 'Error: passed in unknown CV Parameter: %s' % cv

    return cv

def prepare_data(train):

    # Get data ready
    train_X = []
    train_Y = []
    for i in xrange(len(train)):
        train_X.append(train[i][FEATURE_LOC])
        train_Y.append(train[i][LABEL_LOC])

    return (train_X, train_Y)

def get_train_ids(train):
    '''
    Only call if we want to do specific kinds of LOOCV

    e.g. remove all examples of bowl or push
    '''
    #train_id = []
    #for key in train:
    #    train_id.append(int(str(train[key][1])+key.split('_')[-1]))
    
    return None

def execute_grid_search(train, cv, model, parameters, n_jobs):
    '''
    This function actually executes the grid search

    Input: train_X - nxm feature vectors to train over
           train_Y - nx1 vector of labels (can be None if unsupervised or HMM) 
           CV - cv type
           model (e.g. LinearSVC, GaussianHMMClassifie)
           parameters - dictionary of parameters (specific to model)
           n_jobs (parallel training - sometimes has race conditions if more than 1)
    
    Output: The grid used to train and the best classifier
    ''' 

    # Prepare data
    train_X, train_Y = prepare_data(train)
    train_id = get_train_ids(train) #later can make this function more specific

    # setup cross validation
    num_train = len(train_X)
    cv = _cv_setup_helper(cv, num_train, indices=train_id)

    grid = GridSearchCV(model,
                        parameters,
                        n_jobs = n_jobs,
                        cv=cv,
                        verbose =DEFAULT_VERBOSE 
                        )
 
    # Fit the data
    grid.fit(train_X, train_Y) 

    # Pull out the best HMM
    best_clf = grid.best_estimator_

    # Create dictionary of things to return
    results = defaultdict(dict)
    results[TRAIN_KEY][FEAT_KEY] = train_X
    results[TRAIN_KEY][LABEL_KEY] = train_Y
    results[CV_KEY] = grid
    results[CLF_KEY] = best_clf

    return results


def train_hmm_gridsearch(train, cv=DEFAULT_N_FOLD, gmm=False, 
                         n_jobs=DEFAULT_JOBS, p=DEFAULT_P,
                         n_components=DEFAULT_N_COMPONENTS,
                         n_iter=DEFAULT_ITER,
                         covariance_type=DEFAULT_COVAR):
    '''
    Main function that does CV to select parameters of 
    the ideal HMM

    Input: train - python dictionary with training values by key
                   Each entry has three values in a list
                        (0) - train feature vector
                        (1) - label (1: positive, 0: negative) #TODO: CHECK
                        (2) - example id (if we want to do LOOCV)
              
           p - number of examples to leave out for Leave P Out Cross Validation

           n_jobs - how many parallel jobs to run   
                    Default: 1 (Note this function is currently broken)

           gmm - True/False (Do you want to use GMM to represent hidden state?
                 Default: False (use just Multivariate Gaussian)

           
    '''
    # Pull out the number of key frames to see if we could use them for num_states
    #n_states = train[train.keys()[0]][2]

    # Parameters to cross-validate over
    parameters = { 
                  'n_iter':n_iter,
                  'n_components':n_components,
                  #'covariance_type':covariance_type,
                  }   

    # If we want to use the GMMHMM classifier or just the vanilla GaussianHMM
    if (gmm):
        clf = GMMHMMClassifier(n_mix=gmm)
        
    else:
        clf = GaussianHMMClassifier(covariance_type='full')

    # Actually execute the search
    return execute_grid_search(train, cv, clf, parameters, n_jobs)

def train_svc_gridsearch(train, cv=DEFAULT_N_FOLD, 
                         n_jobs=DEFAULT_JOBS, p=DEFAULT_P,
                         C=DEFAULT_SVM_C,
                         kernel=DEFAULT_SVM_KERNELS,
                         penalty=DEFAULT_SVM_PENALTY,
                         class_weight=DEFAULT_SVM_WEIGHT,
                         probability=DEFAULT_SVM_PROB
                         ):
    '''
    Main function that does CV to select parameters of 
    the ideal SVC - Support Vector Classification

    This allows for more options for kernels, but less options
    for linear classification. See linearSVC method for more linear options

    Input: train - python dictionary with training values by key
                   Each entry has three values in a list
                        (0) - train feature vector
                        (1) - label (1: positive, 0: negative) #TODO: CHECK
                        (2) - example id (if we want to do LOOCV)
          
        See http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
        for parameter option information
 
    '''
    # Parameters to cross-validate over
    parameters = { 
                  'C': C,
                  'kernel': kernel,
                  'penalty': penalty,
                  'class_weight': class_weight
                  }   

    clf = sklearn.svm.SVC(probability=probability)

    # Actually execute the search
    return execute_grid_search(train, cv, clf, parameters, n_jobs)

def train_linearsvc_gridsearch(train, cv=DEFAULT_N_FOLD, 
                         n_jobs=DEFAULT_JOBS, p=DEFAULT_P,
                         C=DEFAULT_SVM_C,
                         loss=DEFAULT_SVM_LOSS,
                         dual=DEFAULT_SVM_DUAL,
                         penalty=DEFAULT_SVM_PENALTY,
                         class_weight=DEFAULT_SVM_WEIGHT
                         ):
    '''
    Main function that does CV to select parameters of 
    the ideal linearSVC - Linear Support Vector Classification

    This allows for more options for lienar SVC

    Input: train - python dictionary with training values by key
                   Each entry has three values in a list
                        (0) - train feature vector
                        (1) - label (1: positive, 0: negative) #TODO: CHECK
                        (2) - example id (if we want to do LOOCV)
              
           p - number of examples to leave out for Leave P Out Cross Validation
          
        See http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        for parameter option information
 
    '''
    # Parameters to cross-validate over
    parameters = { 
                  'C': C,
                  'loss': loss,
                  'dual': dual,
                  'penalty': penalty,
                  'class_weight': class_weight
                  }   

    clf = sklearn.svm.LinearSVC()

    # Actually execute the search
    return execute_grid_search(train, cv, clf, parameters, n_jobs)

