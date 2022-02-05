import numpy as np
from scipy import stats

np.random.seed(123)

def data_generation_by_noise_level(noise_std_stepsize_ratio, num_sample=30, num_shift=3, size=1000, rng_seed=123):
    rng = np.random.RandomState(seed=rng_seed)

    # preserve stepsize; change noise/stepsize ratio
    # preserve number of samples, number of shifts, and starting point to be 0.0
    size_dataset = size
    stepsize = 1.0
    noise_std_stepsize_ratio = noise_std_stepsize_ratio
    noise_std = stepsize * noise_std_stepsize_ratio

    dataset = np.zeros((size_dataset, num_sample))
    target = np.zeros((size_dataset, num_sample))
    for i in range(size_dataset):
        shift_points = np.concatenate(([0], np.sort(rng.choice(range(1, num_sample), num_shift)), [num_sample]))
        shift_lens = shift_points[1:] - shift_points[:-1]
        shifts = np.concatenate((rng.choice([-stepsize, stepsize], num_shift), [0]))
        starting_level = 0.0
        loc = []
        for (l, s) in zip(shift_lens, shifts):
            loc += [starting_level]*l
            starting_level += s
        xs = rng.normal(loc=loc, scale=noise_std)
        target[i,:] = loc
        dataset[i,:] = xs
        
    expanded_target = np.concatenate((stepsize * np.ones((size_dataset, 1)), target), axis=1)
    state = np.abs(expanded_target[:,1:] - expanded_target[:,:-1]) / stepsize
    return dataset, target, state


def variational_beam_search(xs, broadening, beam_size, noise_std, prior_variance=9999.0, bias=0.0, verbose=False):
    '''Conditional conjugate model with spike-and-slab Gaussian prior and 
    Gaussian likelihood for 1-dimensional case. 
    (It can be expanded into high-dimensional case.)

    Beam search is used for posterior model selection.

    We will assume the generative model is the true generative model.

    Parameters:
    xs -- numpy array of shape (L,)
    broadening -- float, diffusion constant
    beam_size -- integer, number of hypothesis
    noise_std -- float, true standard deviation of noise
    prior_variance -- float, variance of the prior Gaussian distribution 
        when `s_t = 1`
    bias -- float, `\\xi` in state prior `p(s_t)`
    verbose -- boolean, if true, history of beam updates will be returned

    Returns:
    beam_mu --  posterior mean, of shape (beam_size, L)
    beam_variance -- posteror variance, of shape (beam_size, L)
    beam_log_s -- log probability of each hypothsis over time, of shape 
        (beam_size, L)
    beam_decision -- states at each time step, of shape (beam_size, L)
    history_beam_mu -- list of `beam_mu` at every time step, of shape 
        (L, beam_size, L)
    history_beam_decision -- list of states at every time step, of shape 
        (L, beam_size, L)
    '''

    # generative model parameter
    likelihood_variance = noise_std**2

    # prior distribution for first time step
    prior_mu = 0.0
    # prior_variance = prior_variance  # probably not so important to tune

    # broadening = brodening  # tune this
    # bias = bias  # tune this

    # beam search params
    ndim = 1
    # beam_size = beam_size
    beam_log_s = np.zeros((beam_size, ndim))
    beam_log_s[0] = 100 # enables the first few steps to propagate correctly
    beam_mu = np.ones((beam_size, ndim))
    beam_variance = np.ones((beam_size, ndim))
    beam_decision = np.ones((beam_size, ndim))
    
    if verbose:
        history_beam_decision = [beam_decision]
        history_beam_mu = [beam_mu]
        history_beam_variance = [beam_variance]
        history_beam_log_s = [beam_log_s]

    for i, x in enumerate(xs):
        if i == 0:
            # fit variational distribution only for single prior N(prior_mu, prior_variance)
            mu = ((prior_variance * x + likelihood_variance * prior_mu) / 
                  (prior_variance + likelihood_variance))
            variance = ((prior_variance * likelihood_variance) /
                        (prior_variance + likelihood_variance))

            beam_mu *= mu
            beam_variance *= variance
        else:
            # transport prior forward in time
            prior_mu_s0 = beam_mu[:, -1]
            prior_variance_s0 = beam_variance[:, -1]
            prior_mu_s1 = prior_variance * beam_mu[:, -1] / (prior_variance + beam_variance[:, -1] + broadening)
            prior_variance_s1 = 1 / (1 / (beam_variance[:, -1] + broadening) + 1 / prior_variance)

            # fit variational distribution for both s=0 and s=1
            q_mu_s0 = ((prior_variance_s0 * x + likelihood_variance * prior_mu_s0) / 
                       (prior_variance_s0 + likelihood_variance))
            q_variance_s0 = ((prior_variance_s0 * likelihood_variance) /
                             (prior_variance_s0 + likelihood_variance))
            q_mu_s1 = ((prior_variance_s1 * x + likelihood_variance * prior_mu_s1) / 
                       (prior_variance_s1 + likelihood_variance))
            q_variance_s1 = ((prior_variance_s1 * likelihood_variance) /
                             (prior_variance_s1 + likelihood_variance))

            # Decide whether s=0 or s=1 is better. Maximize ELBO over s
            # make sure to include "constants"
            elbo_s0 = (-(q_variance_s0 + (q_mu_s0 - x)**2) / (2 * likelihood_variance) 
                      -(q_variance_s0 + (q_mu_s0 - prior_mu_s0)**2) / (2 * prior_variance_s0)
                      + (np.log(q_variance_s0) - np.log(likelihood_variance) - np.log(prior_variance_s0)) / 2)
            elbo_s1 = (-(q_variance_s1 + (q_mu_s1 - x)**2) / (2 * likelihood_variance) 
                      -(q_variance_s1 + (q_mu_s1 - prior_mu_s1)**2) / (2 * prior_variance_s1)
                      + (np.log(q_variance_s1) - np.log(likelihood_variance) - np.log(prior_variance_s1)) / 2)

            q_log_s1 = -np.log1p(np.exp(-(elbo_s1 - elbo_s0 + bias)))
            q_log_s0 = -np.log1p(np.exp(elbo_s1 - elbo_s0 + bias))

            # beam search
            candidate_beam_log_s = np.tile(beam_log_s[:, -1], 2) + np.squeeze(np.concatenate((q_log_s1, q_log_s0)))
            ind = np.argsort(-candidate_beam_log_s)[:beam_size]
            extended_beam_log_s = np.tile(beam_log_s, (2, 1))
            beam_log_s = np.concatenate((extended_beam_log_s[ind, :], np.expand_dims(candidate_beam_log_s[ind], 1)), axis=-1)

            extended_candidate_mu = np.expand_dims(np.concatenate((q_mu_s1, q_mu_s0)), 1)
            extended_beam_mu = np.tile(beam_mu, (2, 1))
            beam_mu = np.concatenate((extended_beam_mu[ind, :], extended_candidate_mu[ind]), axis=-1)
            extended_candidate_variance = np.expand_dims(np.concatenate((q_variance_s1, q_variance_s0)), 1)
            extended_beam_variance = np.tile(beam_variance, (2, 1))
            beam_variance = np.concatenate((extended_beam_variance[ind, :], extended_candidate_variance[ind]), axis=-1)
            extended_candidate_decision = np.expand_dims(np.repeat([1, 0], beam_size), 1)
            extended_beam_decision = np.tile(beam_decision, (2, 1))
            beam_decision = np.concatenate((extended_beam_decision[ind, :], extended_candidate_decision[ind]), axis=-1)
            
            if verbose:
                history_beam_mu.append(beam_mu)
                history_beam_decision.append(beam_decision)
                history_beam_variance.append(beam_variance)
                history_beam_log_s.append(beam_log_s)

    beam_log_s -= 100
    
    if verbose:
        return beam_mu, beam_variance, beam_log_s, beam_decision, history_beam_mu, history_beam_variance, history_beam_log_s, history_beam_decision
    else:
        return beam_mu, beam_variance, beam_log_s, beam_decision, None, None, None, None


def exact_beam_search(xs, broadening, beam_size, noise_std, prior_variance=9999.0, bias=0.0, verbose=False):
    '''
    ***THE exact_beam_search IS THE SAME AS variational_beam_search FOR THIS TRACTABLE TOY EXAMPLE.***

    Conditional conjugate model with spike-and-slab Gaussian prior and 
    Gaussian likelihood for 1-dimensional case. 
    (It can be expanded into high-dimensional case.)

    Beam search is used for posterior model selection.

    We will assume the generative model is the true generative model.

    Parameters:
    xs -- numpy array of shape (L,)
    broadening -- float, diffusion constant
    beam_size -- integer, number of hypothesis
    noise_std -- float, true standard deviation of noise
    prior_variance -- float, variance of the prior Gaussian distribution 
        when `s_t = 1`
    bias -- float, `\\xi` in state prior `p(s_t)`
    verbose -- boolean, if true, history of beam updates will be returned

    Returns:
    beam_mu --  posterior mean, of shape (beam_size, L)
    beam_variance -- posteror variance, of shape (beam_size, L)
    beam_log_s -- log probability of each hypothsis over time, of shape 
        (beam_size, L)
    beam_decision -- states at each time step, of shape (beam_size, L)
    history_beam_mu -- list of `beam_mu` at every time step, of shape 
        (L, beam_size, L)
    history_beam_decision -- list of states at every time step, of shape 
        (L, beam_size, L)
    '''

    # generative model parameter
    likelihood_variance = noise_std**2

    # prior distribution for first time step
    prior_mu = 0.0
    # prior_variance = prior_variance  # probably not so important to tune

    # broadening = brodening  # tune this
    # bias = bias  # tune this

    # beam search params
    ndim = 1
    # beam_size = beam_size
    beam_log_s = np.zeros((beam_size, ndim))
    beam_log_s[0] = 100 # enables the first few steps to propagate correctly
    beam_mu = np.ones((beam_size, ndim))
    beam_variance = np.ones((beam_size, ndim))
    beam_decision = np.ones((beam_size, ndim))
    
    if verbose:
        history_beam_decision = [beam_decision]
        history_beam_mu = [beam_mu]
        history_beam_variance = [beam_variance]
        history_beam_log_s = [beam_log_s]

    for i, x in enumerate(xs):
        if i == 0:
            # fit variational distribution only for single prior N(prior_mu, prior_variance)
            mu = ((prior_variance * x + likelihood_variance * prior_mu) / 
                  (prior_variance + likelihood_variance))
            variance = ((prior_variance * likelihood_variance) /
                        (prior_variance + likelihood_variance))

            beam_mu *= mu
            beam_variance *= variance
        else:
            # transport prior forward in time
            prior_mu_s0 = beam_mu[:, -1]
            prior_variance_s0 = beam_variance[:, -1]
            prior_mu_s1 = prior_variance * beam_mu[:, -1] / (prior_variance + beam_variance[:, -1] + broadening)
            prior_variance_s1 = 1 / (1 / (beam_variance[:, -1] + broadening) + 1 / prior_variance)

            # fit variational distribution for both s=0 and s=1
            post_mu_s0 = ((prior_variance_s0 * x + likelihood_variance * prior_mu_s0) / 
                       (prior_variance_s0 + likelihood_variance))
            post_variance_s0 = ((prior_variance_s0 * likelihood_variance) /
                             (prior_variance_s0 + likelihood_variance))
            post_mu_s1 = ((prior_variance_s1 * x + likelihood_variance * prior_mu_s1) / 
                       (prior_variance_s1 + likelihood_variance))
            post_variance_s1 = ((prior_variance_s1 * likelihood_variance) /
                             (prior_variance_s1 + likelihood_variance))

            # Decide whether s=0 or s=1 is better
            # conditional posterior predictive distribution
            post_pred_mu_s0 = prior_mu_s0
            post_pred_variance_s0 = prior_variance_s0 + likelihood_variance
            post_pred_mu_s1 = prior_mu_s1
            post_pred_variance_s1 = prior_variance_s1 + likelihood_variance

            log_post_pred_s0 = (-np.log(post_pred_variance_s0) / 2 
                                - (x - post_pred_mu_s0)**2 / (2 * post_pred_variance_s0))
            log_post_pred_s1 = (-np.log(post_pred_variance_s1) / 2 
                                - (x - post_pred_mu_s1)**2 / (2 * post_pred_variance_s1))

            log_p_s1 = -np.log1p(np.exp(-(log_post_pred_s1 - log_post_pred_s0 + bias)))
            log_p_s0 = -np.log1p(np.exp(log_post_pred_s1 - log_post_pred_s0 + bias))

            # beam search
            candidate_beam_log_s = np.tile(beam_log_s[:, -1], 2) + np.squeeze(np.concatenate((log_p_s1, log_p_s0)))
            ind = np.argsort(-candidate_beam_log_s)[:beam_size]
            extended_beam_log_s = np.tile(beam_log_s, (2, 1))
            beam_log_s = np.concatenate((extended_beam_log_s[ind, :], np.expand_dims(candidate_beam_log_s[ind], 1)), axis=-1)

            extended_candidate_mu = np.expand_dims(np.concatenate((post_mu_s1, post_mu_s0)), 1)
            extended_beam_mu = np.tile(beam_mu, (2, 1))
            beam_mu = np.concatenate((extended_beam_mu[ind, :], extended_candidate_mu[ind]), axis=-1)
            extended_candidate_variance = np.expand_dims(np.concatenate((post_variance_s1, post_variance_s0)), 1)
            extended_beam_variance = np.tile(beam_variance, (2, 1))
            beam_variance = np.concatenate((extended_beam_variance[ind, :], extended_candidate_variance[ind]), axis=-1)
            extended_candidate_decision = np.expand_dims(np.repeat([1, 0], beam_size), 1)
            extended_beam_decision = np.tile(beam_decision, (2, 1))
            beam_decision = np.concatenate((extended_beam_decision[ind, :], extended_candidate_decision[ind]), axis=-1)
            
            if verbose:
                history_beam_mu.append(beam_mu)
                history_beam_decision.append(beam_decision)
                history_beam_variance.append(beam_variance)
                history_beam_log_s.append(beam_log_s)

    beam_log_s -= 100
    
    if verbose:
        return beam_mu, beam_variance, beam_log_s, beam_decision, history_beam_mu, history_beam_variance, history_beam_log_s, history_beam_decision
    else:
        return beam_mu, beam_variance, beam_log_s, beam_decision, None, None, None, None


def delayed_update_procedure(beam_decision, beam_mu, beam_variance, pos_interest):
    '''A procedure that performs delayed update mentioned in the paper.

    Parameters:
    beam_decision -- states of shape (beam_size, L)
    beam_mu -- variational posterior mean of shape (beam_size, L)
    beam_variance -- variational posterior variance of shape (beam_size, L)
    pos_interest -- integer, which hypothesis of interest

    Returns:
    Decisions of shape (L,) and variational parameters of shape (2, L) where 
        `mu` and `variance` are stacked.
    '''
    decisions = beam_decision[pos_interest,:]
    variational_params = np.concatenate((np.expand_dims(beam_mu[pos_interest,:], 1), 
                                        np.expand_dims(beam_variance[pos_interest,:], 1)), axis=1)
    stay_start = 0
    for i, (d, (mu, var)) in enumerate(zip(decisions, variational_params)):
        if d == 1 and i > 0:
            variational_params[stay_start:i, :] = [variational_params[i-1]] * (i - stay_start)
            stay_start = i
    variational_params[stay_start:, :] = [variational_params[-1]] * (len(decisions) - stay_start)

    return decisions, variational_params


def confusion_matrix(truth, prediction):
    ''' both truth and prediction are of shape (L,)
    '''
    tp = 1.0 * np.sum(np.logical_and(truth == 1, prediction == 1)) / np.sum(truth == 1)
    tn = 1.0 * np.sum(np.logical_and(truth == 0, prediction == 0)) / np.sum(truth == 0)
    fp = 1.0 * np.sum(np.logical_and(truth == 0, prediction == 1)) / np.sum(truth == 0)
    fn = 1.0 * np.sum(np.logical_and(truth == 1, prediction == 0)) / np.sum(truth == 1)
    return tp, tn, fp, fn


def pred_log_likelihood(xs, beam_log_s, beam_mu, beam_variance):
    '''DEPRECATED: can be negative infinity because of underflow in logrithm
    '''
    likelihood = 0.0
    prob_s = np.exp(beam_log_s)
    weights = prob_s / np.sum(prob_s, axis=0, keepdims=True)
    for (x, mu, var, weight) in zip(xs[1:], beam_mu[:,:-1].transpose(), beam_variance[:,:-1].transpose(), weights.transpose()):
        likelihood += np.log(np.sum(weight * stats.norm.pdf(x, loc=mu, scale=np.sqrt(var))))
    likelihood /= len(xs)
    return likelihood


def pred_likelihood(xs, beam_log_s, beam_mu, beam_variance):
    '''
    '''
    likelihood = 0.0
    prob_s = np.exp(beam_log_s)
    weights = prob_s / np.sum(prob_s, axis=0, keepdims=True)
    for (x, mu, var, weight) in zip(xs[1:], beam_mu[:,:-1].transpose(), beam_variance[:,:-1].transpose(), weights[:,:-1].transpose()):
        likelihood += np.sum(weight * stats.norm.pdf(x, loc=mu, scale=np.sqrt(var)))
    likelihood /= len(xs)
    return likelihood

def posterior_pred_likelihood(xs, noise_std, broadening, history_beam_log_s, history_beam_mu, history_beam_variance, prior_variance=9999.0, bias=0.0):
    '''Same parameter setting with conjugate_beam_search()
    '''
    beam_log_s = np.zeros_like(history_beam_log_s[-1])
    for (i, term) in enumerate(history_beam_log_s):
        if i != len(history_beam_log_s) - 1:
            beam_log_s[:, i] = term[:, -1] - 100
        else:
            beam_log_s[:, i] = term[:, -1]
    beam_mu = np.zeros_like(history_beam_mu[-1])
    for (i, term) in enumerate(history_beam_mu):
        beam_mu[:, i] = term[:, -1]
    beam_variance = np.zeros_like(history_beam_variance[-1])
    for (i, term) in enumerate(history_beam_variance):
        beam_variance[:, i] = term[:, -1]

    likelihood_variance = noise_std**2

    likelihood = 0.0
    likelihood_set = []
    prob_s = np.exp(beam_log_s)
    weights = prob_s / np.sum(prob_s, axis=0, keepdims=True)
    for (x, mu, var, weight) in zip(xs[1:], beam_mu[:,:-1].transpose(), beam_variance[:,:-1].transpose(), weights[:,:-1].transpose()):
        # transport prior forward in time
        prior_mu_s0 = mu
        prior_variance_s0 = var
        prior_mu_s1 = prior_variance * mu / (prior_variance + var + broadening)
        prior_variance_s1 = 1 / (1 / (var + broadening) + 1 / prior_variance)

        post_pred_mu_s0 = prior_mu_s0
        post_pred_variance_s0 = prior_variance_s0 + likelihood_variance
        post_pred_mu_s1 = prior_mu_s1
        post_pred_variance_s1 = prior_variance_s1 + likelihood_variance

        post_pred_likelihood_s0 = stats.norm.pdf(x, loc=post_pred_mu_s0, scale=np.sqrt(post_pred_variance_s0))
        post_pred_likelihood_s1 = stats.norm.pdf(x, loc=post_pred_mu_s1, scale=np.sqrt(post_pred_variance_s1))

        p_s1 = 1.0 / (1 + np.exp(-bias))
        p_s0 = 1 - p_s1
        post_pred_likelihood = p_s0 * post_pred_likelihood_s0 + p_s1 * post_pred_likelihood_s1

        _likelihood = np.sum(weight * post_pred_likelihood)
        likelihood_set.append(post_pred_likelihood)
        likelihood += _likelihood
    likelihood /= (len(xs) - 1)
    return likelihood, np.array(likelihood_set).transpose(), weights, prob_s
