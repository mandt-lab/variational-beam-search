import os
import sys
import numpy as np
from scipy import stats


class BayesLinReg:
    ''' y = w*x + eps where eps is iid Gaussian noise and w has its prior distribution
    
    Note we compute the posterior mean explicitly but not the covariance explicitly.
    '''
    def __init__(self, num_feature, sigma_n, mu_0=None, sigma_p=None, Lambda=None):
        '''
        Arguments:
        num_feature -- a scalar with the bias term
        sigma_n -- a scalar for the noise variance
        mu_0 -- 1D array that has the length of `num_feature`
        sigma_p -- a 2D array for the covariance matrix
        Lambda -- a 2D array for the precision matrix
        '''
        self.num_feature = num_feature
        # moments parameterization
        if mu_0 is not None:
            self.mu = mu_0
        else:
            self.mu = np.zeros(num_feature)
        self.sigma_n = sigma_n # noise variance, a scalar
        
        self.Sigma = None
        # natural parameterization
        if sigma_p is not None:
            self.Sigma = sigma_p # prior variance, a matrix
            self.Lambda = np.linalg.inv(self.Sigma)
        elif Lambda is not None:
            self.Lambda = Lambda
        else:
            raise("At least one of sigma_p and Lambda is not None.")
        self.eta = np.dot(self.Lambda, self.mu)
        
    def predict_mean(self, x):
        return np.matmul(self.mu, x)
    
    def predict_var(self, x, obs_noise=True):
        if self.Sigma is None:
            self._compute_cov()
        if obs_noise:
            sigma = np.sum(x * np.dot(self.Sigma, x), axis=0) + self.sigma_n
        else:
            sigma = np.sum(x * np.dot(self.Sigma, x), axis=0)
        return sigma
    
    def marginal_likelihood(self, x, y, obs_noise=True):
        return stats.norm.pdf(y, loc=self.predict_mean(x), scale=np.sqrt(self.predict_var(x, obs_noise=obs_noise)))
    
    def log_marginal_likelihood(self, x, y, obs_noise=True):
        return stats.norm.logpdf(y, loc=self.predict_mean(x), scale=np.sqrt(self.predict_var(x, obs_noise=obs_noise)))
    
    def logit_predict_map(self, x):
        '''Logistic function of the MAP of the Gaussian function.
        '''
        logodd = np.matmul(self.mu, x)
        return 1 / (1 + np.exp(-logodd))
    
    def mean_logitnormal_approx(self, x):
        # https://github.com/tensorflow/probability/blob/v0.11.1/tensorflow_probability/python/distributions/logitnormal.py#L90-L126
        # Although it can be implemented it should be the same with `self.logit_predict_map` by Bishop et al., 1995 Neural Networks for Pattern Recognition Section 10.3
        _mu = self.predict_mean(x)
        _sigma = self.predict_var(x) # eigendecomposition of the precision matrix
        b2 = (np.pi / 8.) * _sigma
        return 1 / (1 + np.exp(-_mu/np.sqrt(1. + b2)))
    
    def broaden_temper(self, beta):
        # update Lambda
        self.Lambda *= beta
        # update eta
        self.eta = np.dot(self.Lambda, self.mu)
        
    def broaden_Bayesian_forget(self, beta):
        '''http://www.lucamartino.altervista.org/2003-003.pdf
        https://openreview.net/forum?id=SJlsFpVtDB
        '''
        # prior
        mu0 = np.zeros(self.num_feature)
        Lambda0 = (1-beta)*np.eye(self.num_feature)
        eta0 = np.dot(Lambda0, mu0)
        # posterior
        Lambda = beta*self.Lambda
        eta = np.dot(Lambda, self.mu)
        # update
        self.Lambda = Lambda0 + Lambda
        self.eta = eta0 + eta
        # be aware that self.mu will be computed in self.update()
        # thus we do not compute mu to save computation
        
    def broaden_directional_forget(self, beta, x):
        ''' https://www.sciencedirect.com/science/article/pii/S0005109800000935
        
        x contains the information about which direction needs to be forgotten
        '''
        scale_x = x / np.sqrt(self.sigma_n)
        dir_info = np.matmul(self.Lambda, scale_x)
        alpha = 1 / np.dot(scale_x, dir_info)
        Lambda2 = alpha*np.outer(dir_info, dir_info)
        Lambda1 = self.Lambda - Lambda2
        self.Lambda = Lambda1 + beta*Lambda2 
        # update eta
        self.eta = np.dot(self.Lambda, self.mu)
        
    def broaden_directional_forget_phi(self, beta, x):
        ''' https://www.sciencedirect.com/science/article/pii/S0005109800000935
        
        It is different from the above one in that it forgets information in the direction of x rather than Lambda*x.
        
        x contains the information about which direction needs to be forgotten
        '''
        self._compute_cov()
        scale_x = x / np.sqrt(self.sigma_n)
        dir_info = np.matmul(self.Sigma, scale_x)
        inv_r = 1 / np.dot(scale_x, dir_info)
        alpha = beta - (1-beta)*inv_r
        self.Lambda -= (1-alpha)*np.outer(scale_x, scale_x)
        # update eta
        self.eta = np.dot(self.Lambda, self.mu)
        
    def broaden_selected_forget(self, beta, x):
        ''' 
        Only broaden the subset of parameters that x affects.
        '''
        idx = np.where(np.abs(x) == 0)[0]
        scale_x = x / np.sqrt(self.sigma_n)
        scale_x[idx] = 0.
        r1_info_mat = np.outer(scale_x, scale_x) # rank-1 information matrix
        broaden_idx = np.where(r1_info_mat != 0)
        self.Lambda[broaden_idx] *= beta
        # update eta
        self.eta = np.dot(self.Lambda, self.mu)

    def _compute_cov(self):
        w, v = np.linalg.eigh(self.Lambda)
        self.Sigma = v @ np.diag(1/w) @ v.transpose()
    
    def update(self, x, y, compute_cov=True):
        self.Lambda = self.Lambda + np.outer(x, x)/self.sigma_n
        self.eta = self.eta + y*x/self.sigma_n
        self.Lambda += (1e-6*np.eye(self.num_feature)) # keep stability
        self.mu = np.linalg.solve(self.Lambda, self.eta)
        # compute covariance
        if compute_cov:
            self._compute_cov()

 