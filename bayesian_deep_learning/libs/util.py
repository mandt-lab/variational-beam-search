import os
import sys
from datetime import datetime
import pickle
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
tfd = tfp.distributions

def print_log(*args, **kwargs):
    print("[{}]".format(datetime.now()), *args, **kwargs)
    sys.stdout.flush()

def save_weights(weights_list, outfile_name='weights.pkl'):
    # save with the binary protocol
    with open(outfile_name, 'wb') as outfile:
        pickle.dump(weights_list, outfile, pickle.HIGHEST_PROTOCOL)

def load_weights(infile_name='weights.pkl'):
    # save with the binary protocol
    print_log('load weights from ' + infile_name)
    with open(infile_name, 'rb') as infile:
        weights_list = pickle.load(infile)
    return weights_list

def _initial_multivariate_normal_fn_wrapper(prior_var):
    print(f"Use zero mean multivariate normal prior with {prior_var} variance.")
    def _initial_multivariate_normal_fn(dtype, shape, name, 
                                        trainable, add_variable_fn):
        del name, trainable, add_variable_fn   # unused
        dist = tfd.Normal(
            loc=tf.zeros(shape, dtype), 
            scale=dtype.as_numpy_dtype(np.sqrt(prior_var)))
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)
    return _initial_multivariate_normal_fn

def ind_multivariate_normal_fn(prior_var=1e-2, mu=None, sigma=None):
    """A closure: return the function used for `kernel_prior_fn`.
    See `https://github.com/tensorflow/probability/blob/v0.11.0
            /tensorflow_probability/python/layers/util.py#L202-L224`
    """
    if mu is not None and sigma is not None:
        assert mu.shape == sigma.shape
    else:
        # use multivariate normal prior with specified prior variance
        return _initial_multivariate_normal_fn_wrapper(prior_var)

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates multivariate `Normal` distribution.
        Args:
        dtype: Type of parameter's event.
        shape: Python `list`-like representing the parameter's event shape.
        name: Python `str` name prepended to any created (or existing)
            `tf.Variable`s.
        trainable: Python `bool` indicating all created `tf.Variable`s should be
            added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
        add_variable_fn: `tf.get_variable`-like `callable` used to create (or
            access existing) `tf.Variable`s.
        Returns:
        Multivariate `Normal` distribution.
        """
        del name, trainable, add_variable_fn   # unused
        assert mu.shape == tuple(shape)
        dist = tfd.Normal(
            loc=dtype.as_numpy_dtype(mu), scale=dtype.as_numpy_dtype(sigma))
        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(
            dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


class LaplacePriorRegularizer(regularizers.Regularizer):

    def __init__(self, strength, mu=0., prec=1.):
        '''If `mu` and `prec` are not provided, standard L2 normalized is used.
        '''
        self.strength = strength
        self.mu = mu
        self.prec = prec

    def __call__(self, x):
        return (self.strength/2) * tf.reduce_sum(
            self.prec * tf.square(x - self.mu))


def broaden_weights(prior_m, prior_s, diffusion, mult_diff=True):
    if mult_diff:
        for i in range(len(prior_s)):
            prior_s[i] *= diffusion
    else:
        for i in range(len(prior_s)):
            prior_s[i] += diffusion
    print_log("Weights are broadened.")
    return (prior_m, prior_s)


# beam diversification
def hypothesis_distance(a, b):
    '''Calcualte a distance metric between two sequences of change points.

    Both `a` and `b` must be arrays of zeros and ones (or `True`s and `False`s)
    of equal length >= 1 and the first entry of both `a` and `b` must be one
    (or `True`).

    The function returns:
      `0.5 * (||a||_1 + ||b||_1) * W(a / ||a||_1, b / ||b||_1)`
    where ||.||_1 denotes one-norm and W(., .) is the Wasserstein distance
    between two probability distributions. Thus, the function calculates
    properly normalized probability distributions based on the binary
    sequences `a` and `b`, calculates their Wasserstein distance, and then
    rescales with the average number of change points in `a` and `b`.

    The Wasserstein distance is calculated with metric `g(t, t') = (t - t')^2`.
    
    Complexity: `O(T)` where `T == len(a) == len(b)` is the number of time steps.
    '''

    assert len(a) == len(b)
    assert len(a) >= 1
    assert a[0] == 1
    assert b[0] == 1
    # Additionally, both `a` and `b` may only contain zeros or ones, but we
    # don't check for that.

    a = a.astype(np.int32, copy=True)
    b = b.astype(np.int32, copy=True)
    norm_a = a.sum()
    norm_b = b.sum()

    # Rescale `a` and `b` such that both have norm `norm_a * norm_b`.
    a *= norm_b
    b *= norm_a

    # Scan through `a` and `b` concurrently and move mass around to keep
    # `abs(excess_a)` as small as possible
    cursor_a = 0
    cursor_b = 0
    distance = 0
    while cursor_a != len(a) and cursor_b != len(a):
        if b[cursor_b] >= a[cursor_a]:
            # Move `a[cursor_a]` from `a` to `b`.
            distance += a[cursor_a] * (cursor_a - cursor_b)**2
            b[cursor_b] -= a[cursor_a]
            cursor_a += 1
        else:
            # Move `b[cursor_b]` from `a` to `b`.
            distance += b[cursor_b] * (cursor_a - cursor_b)**2
            a[cursor_a] -= b[cursor_b]
            cursor_b += 1

    # Return a rescaled (floating point) variant of the distance that undoes the initial scaling
    # by `norm_a * norm_b` and then multiplies with the average of `norm_a` and `norm_b`.
    return 0.5 * (1.0 / norm_b + 1.0 / norm_a) * distance

def hamming_distance(a, b):
    return np.sum(np.abs(a-b))

def reject_probability(x):
    '''Return the probability of rejecting to select hypotheses from the same 
    parent.

    The distribution follows Weibull distribution with `lambda = 10` and `k=5`.
    As time goes, the probability increases.
    '''
    assert x >= 0
    return 1 - np.exp(-(x/10)**5)

def is_reject(x, rng):
    rej_prob = reject_probability(x)
    if rng.uniform() < rej_prob:
        return True
    else:
        return False

def is_one_parent_dominate(best_selection):
    '''Utilizing the fact that candidates are arranged with 
        [parent1_0, parent1_1, parent2_0, parent2_1, ...]
    '''
    parents = best_selection // 2
    if len(set(parents)) < len(parents):
        return True
    return False

def beam_diversity(beam):
    '''Calcualate the diversity measure of the hypotheses in the given beam.
    
    This function will probably not be needed for Variational Beam Search.
    It is exposed only for completeness and for debugging. You probably want
    to call `find_optimal_beam` instead.
    
    The argument `beam` must be a numpy tensor of shape `(K, T)` where `K >= 2`
    is the beam size and `T >= 1` is the number of time steps. Each row must be
    a sequence of `T` zeros and ones (or `True`s and `Falses`) with the
    first entry always beeing a one (or `True`). Further, all rows of
    `beam` must be different from each other.
    
    Returns the diversity score (higher means more diverse).
    
    Complexity: `O(K**2 * T)`
    '''
    
    K, T = beam.shape
    assert K >= 2
    assert T >= 1
    
    return np.log([
        hypothesis_distance(beam[i], beam[j]) for i in range(K) for j in range(i)
    ]).sum()

def maximize_diversity(candidates, 
                       beam_size, 
                       individual_scores, 
                       diversity_importance, 
                       rng=None):
    '''Return the `beam_size` optimal of `candidates`.

    Maximizes the trade-off between diversity among hypotheses and individual
    scores of each hypothesis.
    
    Args:
        candidates: tensor of shape `(N, T)` where `N` is the number of
            candidates and `T >= 1` is the number of time steps. Each row must be a
            sequence of `T` zeros and ones (or `True`s and `Falses`) with the first
            entry always beeing a one (or `True`). Further, all rows must be
            different from each other.
        beam_size: the number of hyptheses that can be selected from
        candidates. If `beam_size >= N` then no truncation is needed and
            the function returns the tensor `[0, 1, ..., N - 1]`.
        individual_scores: real valued tensor of shape `(N,)`. Each entry
            describes an individual "importance" of each hypothesis, e.g., its
            posterior log-probability. See objective function below.
        diversity_importance: positive scalar that controls the trade-off
            between diversity and individual scores. See objective function below.

    Returns:
        A tuple `(indices, diversity)`. Here, `indices` is an integer tensor of
        size `min(beam_size, N)` of pairwise disjunct indices into the rows of
        `candidates`. Further, `diversity = beam_diversity(candidates[diversity])`.
        
        The tensor `indices` maximizes the following objective function:

        `objective = (
            individual_scores[ret].sum() +
            diversity_importance * beam_diversity(candidates[ret, :])`
    
    Complexity: `O(N**2 * T + 2**N)`, where the first term comes from calculating
        all pairwise distances and the second term comes from trying out all
        combinations.
    '''
    
    N, T = candidates.shape
    assert T >= 1
    
    if beam_size >= N:
        # No truncation necessary, all candidates fit into the beam.
        return np.arange(N), 0

    if rng is None:
        rng = np.random.RandomState(2**31)
    
    # Calculate all pairwise log-distances (and fill up with zeros
    # so that it has no effect when taking the sum).
    pairwise_dists = np.array([[
        hamming_distance(candidates[i], candidates[j]) if i > j else 0
        for i in range(N)] for j in range(N)])
    
    best_selection = None
    best_score = float('-Inf')
    diversity = None
    while True:
        for selection in itertools.combinations(range(N), beam_size):
            selection = np.array(selection)
            current_diversity = pairwise_dists[selection[:, None], selection[None, :]].sum()
            # Normalize the diversity such that it scales to the situation of 
            # `beam_size`=2, where the number of hypothesis is 2 and the number 
            # of pairwise distance is 1.
            # Thus `diversity_importance` applies for different `beam_size`.
            current_diversity /= (beam_size - 1)
            score = individual_scores[selection].sum() + diversity_importance * current_diversity
            if score > best_score:
                best_score = score
                best_selection = selection
                diversity = current_diversity * (beam_size - 1)
        if is_one_parent_dominate(best_selection):
            print_log("One parent tries to dominate:")
            if is_reject(T, rng):
                # reject this dominate
                # increase `diversity_importance` and try again
                diversity_importance *= 1.2
                print_log("\tReject and new `diversity_importance` is ", 
                          diversity_importance)
            else:
                print_log("\tAgree and current task id: ", T)
                break
        else:
            break

    return best_selection, diversity

def find_optimal_beam(scores, beam_size, discard_fraction = 1.0 / 3.0):
    '''Return the indices of the `beam_size` optimal hypotheses.

    Args:
        scores: vector of scores (e.g., log probabilities or ELBOs) of each
            hypothesis. Must have an even length and the two hypotheses with the
            same parent always have to come together, i.e.,
            scores = [
                score of the first child of the first parent,
                score of the second child of the first parent,
                score of the first child of the second parent,
                score of the second child of the second parent,
                score of the first child of the third parent,
                score of the second child of the third parent,
            ]
        beam_size: the number of hyptheses that can be selected from candidates.
            
        discard fraction: fraction of the lowest scroed hypotheses that will be
            discarded before we even try to maximize diversity. More precisely,
            this is the fraction that will be discarded *in the steady state*,
            i.e., once `len(scores) == 2 * beam_size`. Must be between 0 and 0.5.

    Returns:
        An array of indices into argument `scores` that defines the optimal beam.
    '''
    assert 0 < discard_fraction
    assert discard_fraction < 0.5
    if beam_size >= len(scores):
        return np.arange(len(scores))
    num_parents = len(scores) // 2
    assert scores.shape == (2 * num_parents,)
    assert num_parents <= beam_size
    
    # Keep track of the hypotheses' parents
    parents = np.array([(i, i) for i in range(num_parents)]).flatten()
    
    # Discard `discard_fraction` of the hypotheses (except that we don't have to discard
    # any hypothesis in the first few steps when there are only few hypotheses)
    num_keep = min(len(scores), round((1.0 - discard_fraction) * (2 * beam_size)))
    candidate_indices = np.argsort(-scores)[:num_keep]
    candidate_scores = scores[candidate_indices]
    candidate_parents = parents[candidate_indices]
    
    # Find out how many different parents are among the candidates (but at most `beam_size`).
    max_num_parents = min(beam_size, len(set(candidate_parents)))
    
    # Out of all ways to choose `beam_size` candidates, consider only the ones with
    # `max_num_parents` different parents, and then take the one with maximum total score.
    best_indices = None
    best_score = float('-Inf')
    resulting_beam_size = min(beam_size, len(candidate_scores))
    for indices in itertools.combinations(range(len(candidate_scores)), resulting_beam_size):
        indices = np.array(np.array(indices))
        if len(set(candidate_parents[indices])) == max_num_parents:
            score = candidate_scores[indices].sum()
            if score > best_score:
                best_indices = indices
                best_score = score

    return candidate_indices[best_indices]