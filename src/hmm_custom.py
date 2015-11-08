import sklearn.hmm
import numpy as np
import string

#change a few functions otherwise it won't work with the grid search
class GMMHMMClassifier(sklearn.hmm.GMMHMM):
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
        
        #newX = [x.ravel() for x in X]
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
class GaussianHMMClassifier(sklearn.hmm.GaussianHMM):
    def __init__(self, n_components=1, covariance_type='diag', startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", means_prior=None, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):

        super(GaussianHMMClassifier,self).__init__(n_components=n_components, 
                                                   startprob=startprob, 
                                                   transmat=transmat, 
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params,
                          covariance_type=covariance_type)

        #self.n_symbols = n_symbols
        #self._covariance_type = covariance_type
        
    def fit(self, X, y=None, **kwargs):    
        if "n_symbols" in kwargs:
            self.n_symbols = kwargs["n_symbols"]
        
        self.transmat_ = None
        self.startprob_ = None        
        
        #newX = [x.ravel() for x in X]
        try:
            #return super(GaussianHMMClassifier, self).fit(newX,**kwargs)
            return super(GaussianHMMClassifier, self).fit(X,**kwargs)
        except ValueError, e:
            print "Error during fit:  "\
                  "Message is: ", e
            import pdb; pdb.set_trace()
            raise
    
    def score(self, X, y=None, **kwargs):   
        #newX = [x.ravel() for x in X]
        score = np.mean([super(GaussianHMMClassifier, self).score(x,**kwargs)
                         for x in X])            
        if np.isnan(score):
            score = np.NINF
        return score
    
    def transform(self, X, y=None, **kwargs):
        """ I know this shouldn't be here, but I need it"""
        return X
    

