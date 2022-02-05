import numpy as np
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from variational_beam_search import *

def toy_beam_search():
    # example of beam search of size 2
    beam_size = 2
    broad = 1.0
    noise_std_stepsize_ratio = 0.5
    p = 3.0 / 30
    prior_bias = np.log(p/(1-p))
    dataset, target, state = data_generation_by_noise_level(
        noise_std_stepsize_ratio=noise_std_stepsize_ratio, 
        size=1, 
        num_sample=30, 
        num_shift=2)
    xs = dataset[0,:]
    truth = target[0,:]
    (beam_mu, 
     beam_variance, 
     beam_log_s, 
     beam_decision, 
     history_beam_mu, 
     history_beam_variance, 
     history_beam_log_s, 
     history_beam_decision) = variational_beam_search(
        xs, 
        bias=prior_bias, 
        broadening=broad, 
        beam_size=beam_size, 
        noise_std=noise_std_stepsize_ratio)

    where_A_dominate = np.where(beam_log_s[0,:] - beam_log_s[1,:] > 0)[0][1]

    plt.figure(figsize=(5.5,1.5))

    for pos_interest in range(beam_size):
        decisions, variational_params = delayed_update_procedure(
            beam_decision, beam_mu, beam_variance, pos_interest)
        variational_mean = variational_params[:, 0]
        variational_std = np.sqrt(variational_params[:, 1])
        plt.step(range(len(decisions))[:where_A_dominate+1], 
                 variational_mean[:where_A_dominate+1],
                 color='C%d' % (1-pos_interest),
                 linewidth=1.3,
                 alpha=1.0,
                 where='post',)
        plt.fill_between(range(len(decisions))[:where_A_dominate+1], 
                         variational_mean[:where_A_dominate+1] + variational_std[:where_A_dominate+1], 
                         variational_mean[:where_A_dominate+1] - variational_std[:where_A_dominate+1],
                         step='post',
                         color='C%d' % (1-pos_interest),
                         alpha=0.3)
        
        if pos_interest == 0:
            label = 'higher prob.'
        else:
            label = 'lower prob.'
        plt.step(range(len(decisions))[where_A_dominate:], 
                 variational_mean[where_A_dominate:],
                 color='C%d' % (pos_interest),
                 linewidth=1.3,
                 alpha=1.0,
                 where='post',
                 label=label)
        plt.fill_between(range(len(decisions))[where_A_dominate:], 
                         variational_mean[where_A_dominate:] + variational_std[where_A_dominate:], 
                         variational_mean[where_A_dominate:] - variational_std[where_A_dominate:],
                         step='post',
                         color='C%d' % (pos_interest),
                         alpha=0.3)

    plt.plot(xs, ls='', marker='x', color='C7', alpha=1.0, label='noisy data')
    plt.step(range(len(decisions)), 
             truth, ls='-', linewidth=1.3, color='k', alpha=1.0, where='post', 
             label='ground truth')

    plt.legend(fontsize=8, loc='lower left')
    plt.xlabel('time step $t$', fontsize=12)
    plt.ylabel('$z_t$', rotation=0, fontsize=15)

    plt.xlim(-1, 29.5)

    ax = plt.gca()
    ax.xaxis.set_label_coords(1.0, -0.07)
    ax.yaxis.set_label_coords(-0.06, 0.85)

    plt.plot([where_A_dominate, where_A_dominate], [-3, 1], 
            ls='--',
            color='C3',
            alpha=0.5)
    plt.text(where_A_dominate, -1 -.2, ('when two hypotheses\nswitch the order'),
            ha='center',
            fontsize=10.5,
            color='C3',
            wrap=True)

    plt.savefig('./toy_beam_search.pdf', bbox_inches='tight')


def toy_greedy_search():
    # example of greedy search
    beam_size = 1
    broad = 1.0
    noise_std_stepsize_ratio = 0.5
    p = 3.0 / 30
    prior_bias = np.log(p/(1-p))
    dataset, target, state = data_generation_by_noise_level(
        noise_std_stepsize_ratio=noise_std_stepsize_ratio, 
        size=1, 
        num_sample=30, 
        num_shift=2)
    xs = dataset[0,:]
    truth = target[0,:]
    (beam_mu, 
     beam_variance, 
     beam_log_s, 
     beam_decision, 
     history_beam_mu, 
     history_beam_variance, 
     history_beam_log_s, 
     history_beam_decision) = variational_beam_search(
        xs, 
        bias=prior_bias, 
        broadening=broad, 
        beam_size=beam_size, 
        noise_std=noise_std_stepsize_ratio)

    plt.figure(figsize=(5.5,1.5))

    pos_interest = 0
    decisions, variational_params = delayed_update_procedure(
        beam_decision, beam_mu, beam_variance, pos_interest)
    variational_mean = variational_params[:, 0]
    variational_std = np.sqrt(variational_params[:, 1])
    plt.step(range(len(decisions)), variational_mean,
             color='C1',
             linewidth=2,
             alpha=1.0,
             where='post',
             label='greedy hypothesis')
    plt.fill_between(range(len(decisions)), 
                     variational_mean + variational_std, 
                     variational_mean - variational_std,
                     step='post',
                     color='C1',
                     alpha=0.3)

    plt.plot(xs, ls='', marker='x', color='C7', alpha=1.0, label='noisy samples')
    plt.step(range(len(decisions)), 
             truth, ls='-', linewidth=1.5, color='k', alpha=1.0, where='post', 
             label='truth')
    plt.legend(fontsize=8)
    plt.xlabel('time step $t$', fontsize=13)
    plt.ylabel('$z$', rotation=0, fontsize=13)
    plt.savefig('./toy_greedy_search.pdf', bbox_inches='tight')



if __name__ == "__main__":
    # VBS
    toy_beam_search()

    # VGS
    toy_greedy_search()