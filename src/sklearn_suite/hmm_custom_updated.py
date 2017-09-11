import hmmlearn.hmm
import numpy as np
import string
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

#change a few functions otherwise it won't work with the grid search
class GMMHMMClassifier(hmmlearn.hmm.GMMHMM):
    def __init__(self, n_components=1, n_mix=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", gmms=None, covariance_type='diag',
                 covars_prior=1e-2, random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):


        super(GMMHMMClassifier,self).__init__(n_components=n_components,
                                              n_mix=n_mix, 
                                              startprob=startprob, 
                                              transmat=transmat,
                 startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                 algorithm=algorithm, gmms=gmms, covariance_type=covariance_type,
                 covars_prior=covars_prior, random_state=random_state, n_iter=n_iter, thresh=thresh,
                 params=params,
                 init_params=init_params)

        self.n_symbols = n_symbols
        self._covariance_type = covariance_type
        
    def fit(self, X, y=None, **kwargs):    
        if "n_symbols" in kwargs:
            self.n_symbols = kwargs["n_symbols"]
        
        self.transmat_ = None
        self.startprob_ = None        
        
        try:
            #return super(GaussianHMMClassifier, self).fit(newX,**kwargs)
            return super(GMMHMMClassifier, self).fit(X,**kwargs)
        except ValueError, e:
            print "Error during fit:  "\
                  "Message is: ", e
            raise
    
    def score(self, X, y=None, **kwargs):   
        #newX = [x.ravel() for x in X]
        score = np.mean([super(GMMHMMClassifier, self).score(x,**kwargs)
                         for x in X])            
        if np.isnan(score):
            score = np.NINF
        return score
    
    def transform(self, X, y=None, **kwargs):
        """ I know this shouldn't be here, but I need it"""
        return X
    



#change a few functions otherwise it won't work with the grid search
class GaussianHMMClassifier(hmmlearn.hmm.GaussianHMM):
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", left_right=False):
        if left_right:
            params="mct"
            init_params="cm"
        super(GaussianHMMClassifier,self).__init__(n_components=n_components, 
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.left_right=left_right
        
    def fit(self, X, y=None, **kwargs):    
        if "n_symbols" in kwargs:
            self.n_symbols = kwargs["n_symbols"]
        
        #self.transmat_ = None
        #self.startprob_ = None

        if self.left_right:
            transmat = np.zeros((self.n_components, self.n_components))
         
            # Left-to-right: each state is connected to itself and its
            # direct successor.
            for i in range(self.n_components):
                if i == self.n_components - 1:
                    transmat[i, i] = 1.0
                else:
                    transmat[i, i] = transmat[i, i + 1] = 0.5
         
            # Always start in first state
            startprob = np.zeros(self.n_components)
            startprob[0] = 1.0

            # Store the values now
            self.transmat_= transmat
            self.startprob_ = startprob
         
        newX = np.concatenate(X)
        lengths = [len(x) for x in X]
        try:
            return super(GaussianHMMClassifier, self).fit(newX,lengths=lengths, **kwargs)
        except ValueError, e:
            print "Error during fit:  "\
                  "Message is: ", e
            #import pdb; pdb.set_trace()
            #raise
    
    def score(self, X, y=None, **kwargs):
        try:
            score = np.mean([super(GaussianHMMClassifier, self).score(x,**kwargs)
                             for x in X])            
        except:
            score=np.NINF
        if np.isnan(score):
            score = np.NINF
        return score
    
    def transform(self, X, y=None, **kwargs):
        """ I know this shouldn't be here, but I need it"""
        return X
    

