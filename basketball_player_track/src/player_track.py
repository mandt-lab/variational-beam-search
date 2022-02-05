import numpy as np
from src.bayes_linear_regression import BayesLinReg

class PlayerTracking:
    def __init__(self, sigma_n2, num_coord=2, mu=None, Lambda=None):
        self.num_coord = num_coord
        self.sigma_n2 = sigma_n2
        # decompile arguments
        mu1, mu2 = None, None
        if mu is not None:
            mu1, mu2 = mu
        Lambda1, Lambda2 = np.eye(num_coord), np.eye(num_coord)
        if Lambda is not None:
            Lambda1, Lambda2 = Lambda
        # first row
        self.w1 = BayesLinReg(num_feature=num_coord, sigma_n=sigma_n2, mu_0=mu1, Lambda=Lambda1)
        # second row
        self.w2 = BayesLinReg(num_feature=num_coord, sigma_n=sigma_n2, mu_0=mu2, Lambda=Lambda2)
        
    def log_marginal_likelihood(self, x_old, x_new, obs_noise=True):
        assert np.shape(x_old) == np.shape(x_new) == (2,)
        logp1 = self.w1.log_marginal_likelihood(x_old, x_new[0], obs_noise=obs_noise)
        logp2 = self.w2.log_marginal_likelihood(x_old, x_new[1], obs_noise=obs_noise)
        return logp1 + logp2
    
    def update(self, x_old, x_new, compute_cov=True):
        assert np.shape(x_old) == np.shape(x_new) == (2,)
        self.w1.update(x_old, x_new[0], compute_cov=compute_cov)
        self.w2.update(x_old, x_new[1], compute_cov=compute_cov)
        
    def predict_mean(self, x_cur):
        return self.w1.predict_mean(x_cur), self.w2.predict_mean(x_cur)
        
    def broaden_Bayesian_forget(self, beta):
        self.w1.broaden_Bayesian_forget(beta)
        self.w2.broaden_Bayesian_forget(beta)
        
    def broaden_temper(self, beta):
        self.w1.broaden_temper(beta)
        self.w2.broaden_temper(beta)
        
    def get_params(self):
        mu = [self.w1.mu, self.w2.mu]
        Lambda = [self.w1.Lambda, self.w2.Lambda]
        return mu, Lambda