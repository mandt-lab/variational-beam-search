import os
import sys
import numpy as np
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from vbs import print_log, BeamSearchHelper, get_beam_search_helper
from bayes_linear_regression import BayesLinReg


### plot functions ###
def plot_linear_function_with_interval(bayeelinreg, ax, obs_noise=False):
    N = 100
    X = np.empty((2, N))
    x = np.linspace(-1, 1, N)
    X[0,:] = x
    X[1,:] = np.ones_like(x)
    y = bayeelinreg.predict_mean(X)
    sigma = np.sqrt(bayeelinreg.predict_var(X, obs_noise))
    
    ax.plot(x, y, label='prediction')
    ax.fill_between(x, (y-sigma), (y+sigma), color='b', alpha=.1)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    
def plot_obs(x, y, ax, T=10):
    for i, (_x, _y) in enumerate(zip(np.flip(x[:]), np.flip(y[:]))):
        if i == 0:
            ax.plot(_x, _y, 'xb', alpha=np.exp(-i/T), label='noisy data')
        else:
            ax.plot(_x, _y, 'xb', alpha=np.exp(-i/T))
    
def plot_original_function(w, b, ax):
    N = 100
    X = np.empty((2, N))
    x = np.linspace(-1, 1, N)
    X[0,:] = x
    X[1,:] = np.ones_like(x)
    y = np.dot([w, b], X)
    ax.plot(x, y, 'C3--', label='ground truth\n(MLE)')


### generate toy data ###
rng = np.random.RandomState(seed=1)

# create M pieces of one-dimension linear regression where each component has N points
M = 2
N = 20
x_limit = 1
sigma_n = 0.1 # constant noise variance
x = 2 * x_limit * rng.random_sample((N*M, 1)) - x_limit # x is in the interval of [0, 10)
x = np.concatenate((x, np.ones((N*M, 1))), axis=1)
print("Shape of x", np.shape(x))
# w = 1 * rng.random_sample(M) * rng.choice([-1, 1], size=M) # weights in (-10, 10) 
# b = 1 * rng.random_sample(M) * rng.choice([-1, 1], size=M) # bias in (-10, 10) 

w = [0.7, -0.7]
b = [-0.5, 0.5]

y = np.zeros(M*N)
for i in range(M):
    print('ground truth weight, bias:', [w[i], b[i]])
    y[N*i : N*(i+1)] = np.matmul(x[N*i : N*(i+1), :], [w[i], b[i]])
    
print('Add noise to y')
y = rng.normal(loc=y, scale=np.sqrt(sigma_n))
print('Shape of y', np.shape(y))


### maximum likelihood estimation as the best estimate ###
w_star = []
b_star = []
for i in range(M):
    reg = LinearRegression(fit_intercept=False).fit(x[N*i : N*(i+1), :], y[N*i : N*(i+1)])
    print(reg.coef_)
    w_star.append(reg.coef_[0])
    b_star.append(reg.coef_[1])


### plot initialization ###
matplotlib.rc('font', size=12) 

N_cand = [20, 25, 30, 35, 40]

fig = plt.figure(figsize=(2.5 * len(N_cand), 5.3))
axes = fig.subplots(2, len(N_cand))
axes[0,2].set_title('VCL [Nguyen et al., 2018]')
axes[1,2].set_title('VGS (proposed)')


### Bayesian online learning without adaptation ###
bayeelinreg = BayesLinReg(num_feature=2, sigma_p=np.eye(2), sigma_n=0.1)

axes_vcl = axes[0,:]
j = 0 # plot indicator
k = 0 # true function indicator
plotted_x, plotted_y = [], []
for i, (_x, _y) in enumerate(zip(x, y)):
    plotted_x.append(_x[0])
    plotted_y.append(_y)
    
    bayeelinreg.update(_x, _y)
    
    if i+1 > N:
        k = 1
    
    if i+1 in N_cand:
        if j == 0:
            axes_vcl[j].set_ylabel(r'$y$', rotation='horizontal', fontsize=16)
        
        plot_linear_function_with_interval(bayeelinreg, axes_vcl[j])
        plot_obs(plotted_x, plotted_y, axes_vcl[j])
        # true function
        plot_original_function(w_star[k], b_star[k], axes_vcl[j])
        
        axes_vcl[j].text(0.78, 0.85, 'N=%d' % (i+1), verticalalignment='center', ha='center')
        
        if j == 0:
            axes_vcl[j].legend(fontsize=10, loc='upper left')
            
        j += 1


### Variational Beam Search ###
axes_vgs = axes[1,:]

beam_size = 1
num_feature = 2
p_set = [0.35] 
beta_set = [3.5]
sigma_n_set = [0.1]

# plot
j = 0 # plot indicator
k = 0 # true function indicator

for p in p_set:
    for beta in beta_set:
        for sigma_n in sigma_n_set:
            folder_name = f'./ckpt_vbs_p{p}_beta{beta}_sigma{sigma_n}_b{beam_size}/'
            prior_logodds = np.log(p/(1-p))
            sigma_p = np.eye(num_feature)
            bsh = get_beam_search_helper(folder_name + '/helper.pkl', 
                                         save_folder=folder_name, 
                                         beam_size=beam_size, 
                                         diffusion=beta, 
                                         jump_bias=prior_logodds)
            
            abs_errs = []
            abs_1st_errs = []
            plotted_x, plotted_y = [], [] # plot
            for task_id, (x_train, y_train) in enumerate(zip(x, y)):
                plotted_x.append(x_train[0])
                plotted_y.append(y_train)
                
                x_test = x_train
                y_test = y_train
                if task_id % 2000 == 0:
                    print('='*75)
                    print_log(f"Start task {task_id}.")

                if task_id <= bsh.task_id:
                    print_log(f"Task {task_id} is already trained. Skip training.")
                    continue

                hypotheses = bsh.get_new_hypotheses_args(x_train, y_train)

                mu_set, Lambda_set, elbos, test_y_preds = [], [], [], []
                num_hypotheses = len(hypotheses)
                for hypothesis in hypotheses:
                    (model_name,
                    s_t,
                    diffusion,
                    mu, 
                    Lambda,
                    x_train,
                    y_train) = hypothesis

                    # compile the args
                    if mu is None:
                        mu = np.zeros(num_feature)
                    if Lambda is None:
                        Lambda = np.eye(num_feature)

                    # initialization
                    linreg = BayesLinReg(num_feature=num_feature, 
                                         sigma_n=sigma_n, 
                                         mu_0=mu, 
                                         Lambda=Lambda)
                    
                    # broaden if required
                    if s_t == 1:
                        linreg.broaden_temper(1./diffusion) # beta = diffusion
                    
                    # compute predictive probability
                    marginal_likelihood = linreg.marginal_likelihood(x_train, y_train)
                    if marginal_likelihood == 0:
                        # computation stability
                        raise('marginal_likelihood == 0')
                        log_marginal_likelihood = -5
                    else:
                        log_marginal_likelihood = np.log(marginal_likelihood)
                    elbos.append(log_marginal_likelihood)
                    
                    # absorb new evidence
                    linreg.update(x_train, y_train)
                    
                    # save weights
                    mu_set.append(linreg.mu)
                    Lambda_set.append(linreg.Lambda)
                    
                    # prediction
                    test_y_preds.append(linreg.predict_mean(x_test))


                bsh.absorb_new_evidence_and_prune_beams(elbos, mu_set, Lambda_set)
                
                # plot
                if task_id+1 > N:
                    k = 1
                if task_id+1 in N_cand:
                    if j == 0:
                        axes_vgs[j].set_ylabel(r'$y$', rotation='horizontal', fontsize=16)
                    axes_vgs[j].set_xlabel(r'$x$', fontsize=16)
                        
                    mu = bsh.current_task_res[0]['mu']
                    Lambda = bsh.current_task_res[0]['Lambda']
                    bayeelinreg = BayesLinReg(num_feature=num_feature, sigma_n=sigma_n, mu_0=mu, Lambda=Lambda)
                    bayeelinreg.Sigma = np.linalg.inv(Lambda) # It does not compute Sigma automatically
                    
                    plot_linear_function_with_interval(bayeelinreg, axes_vgs[j])

                    plot_obs(plotted_x, plotted_y, axes_vgs[j])
                    # true function
                    plot_original_function(w_star[k], b_star[k], axes_vgs[j])

                    axes_vgs[j].text(0.78, 0.85, 'N=%d' % (task_id+1), verticalalignment='center', ha='center')

                    j += 1

                # joint probability
                test_y_pred = bsh.weighted_test_probability(test_y_preds)
                test_y_pred = 1 / (1 + np.exp(-test_y_pred))
                abs_err = np.abs(test_y_pred - y_test)
                abs_errs.append(abs_err)
                # most likely probability
                test_y_pred_1st = test_y_preds[bsh._indices[0]]
                test_y_pred_1st = 1 / (1 + np.exp(-test_y_pred_1st))
                abs_err_1st = np.abs(test_y_pred_1st - y_test)
                abs_1st_errs.append(abs_err_1st)
            
                
            print("Average absolute error:", np.mean(abs_errs))
            print("Average absolute error of the most likely hypothesis:", np.mean(abs_1st_errs))

fig.tight_layout()
fig.subplots_adjust(wspace=0.27)
fig.savefig(fname='catastrophic_remembering.pdf')