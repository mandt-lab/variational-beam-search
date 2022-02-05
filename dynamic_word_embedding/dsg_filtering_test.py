import argparse
import pickle as pkl
import numpy as np
import scipy
import tensorflow as tf

from lib.dataset import SingleTimestep


parser = argparse.ArgumentParser(description='''
    Validation process of filtering algorithm for dynamic word embeddings.''')
parser.add_argument('--vocab_size', metavar='N', type=int,
                    help='vocabulary size')
parser.add_argument('--dim', metavar='N', type=int,
                    help='embedding dimension')
parser.add_argument('--rand_vocab', action='store_true', 
    help='''Randomly select `max_vocab_size` words as vocabulary instead of top 
    `max_vocab_size` words. This may lead to less time consuming if you do not
    care too much the embedding quality.''')
parser.add_argument('--next_year', action='store_true',
                    help='validate on next year')
parser.add_argument('--mu_path', metavar='MU_PATH',
                    help='`beam_mu` path.')
parser.add_argument('--log_s_path', metavar='LOG_S_PATH',
                    help='`beam_log_s` path.')
parser.add_argument('--context_vector_ckpt', metavar='FILE_PATH', help='''
        Path to fixed context vectors `V` in the format of TensorFlow checkpoint. The name
        follows `q/mean_v`''')
parser.add_argument('--rng_seed', metavar='N', type=int,
                    help='random number generator seed.')
parser.add_argument('val_data_path', metavar='VAL_DATA_PATH',
                    help='validation data path.')
parser.add_argument('out_file_path', metavar='FILE_PATH',
                    help='Likelihood path.')


class ValidationModel:
    def __init__(self, vocab_size, emb_dimension):
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        self._get_placeholders()
        self.log_likelihood = -self._negative_log_likelihood(self.n_sum, self.counts_pos, self.scaled_freq_neg, self.u, self.v)
        self.sess = tf.Session()

    def pred_likelihood(self, dat, u, v, weight, balancing_constant):
        ''' It returns an array of log-likelihood of each word in shape (V,).

        w(s_{i,1:t}) * p(n_i|u_i, s_{i,1:t})
        '''
        log_likelihood_i = self.sess.run(self.log_likelihood, 
                        feed_dict={
                                    self.n_sum: dat.n_sum,
                                    self.counts_pos: dat.counts_pos,
                                    self.scaled_freq_neg: dat.scaled_freq_neg,
                                    self.u: u,
                                    self.v: v
                                   })

        if balancing_constant is None:
            balancing_constant = -log_likelihood_i

        return weight * np.exp(balancing_constant + log_likelihood_i), balancing_constant

    def pred_log_likelihood(self, dat, u, v, weight):
        ''' It returns an array of log-likelihood of each word in shape (V,).

        w(s_{i,1:t}) * log p(n_i|u_i, s_{i,1:t})
        '''
        return weight * self.sess.run(self.log_likelihood, 
                        feed_dict={
                                    self.n_sum: dat.n_sum,
                                    self.counts_pos: dat.counts_pos,
                                    self.scaled_freq_neg: dat.scaled_freq_neg,
                                    self.u: u,
                                    self.v: v
                                   })

    def _get_placeholders(self):
        self.n_sum = tf.placeholder(tf.float32, shape=(self.vocab_size, self.vocab_size))
        self.counts_pos = tf.placeholder(tf.float32, shape=(self.vocab_size,)) 
        self.scaled_freq_neg = tf.placeholder(tf.float32, shape=(self.vocab_size,))
        self.u = tf.placeholder(tf.float32, shape=(self.vocab_size, self.emb_dimension)) 
        self.v = tf.placeholder(tf.float32, shape=(self.vocab_size, self.emb_dimension)) 

    def _negative_log_likelihood(self, n_sum, counts_pos, scaled_freq_neg, u, v):
        '''Return the negative log likelihood of `dat` (represented by `n_sum`, 
        `counts_pos`, and `scaled_freq_neg`) given `u` and `v` (up to a const offset).
        '''
        with tf.variable_scope('log_likelihood'):
            # For better efficiency, we rewrite
            #   `- log p(n^{+-} | U, V) = - n^+ log(sigmoid(u*v)) - n^- log(sigmoid(-u*v) `
            # as
            #   `- (n^+ + n^-) log(sigmoid(u*v)) + n^- u*v`.

            # First term: `(n^+ + n^-) log(sigmoid(u*v))`
            minus_uv = tf.matmul(-u, v, transpose_b=True) # shape (V, V)
            minus_log_sigmoid_uv = tf.nn.softplus(
                minus_uv, 'minus_log_sigmoid_uv')
            minus_log_likelihood_1 = tf.reduce_sum(n_sum * minus_log_sigmoid_uv, 
                          axis=1, name='minus_log_likelihood_1') # shape (V,)

            # Second term: `n^- u*v`
            # (use `n^-[i, j] = dat.counts_pos[i] * dat.scaled_freq_neg[j]`)
            scaled_v = tf.tensordot(
                scaled_freq_neg, v, 1, 'scaled_v')  # shape (d,)
            scaled_uv = tf.tensordot(
                u, scaled_v, 1, 'scaled_uv')  # shape (V,)
            minus_log_likelihood_2 = scaled_uv * counts_pos # shape (V,)

            return minus_log_likelihood_1 + minus_log_likelihood_2


if __name__ == '__main__':
    args = parser.parse_args()

    out_file = open(args.out_file_path, 'a')

    val_model = ValidationModel(args.vocab_size, args.dim)

    if args.rand_vocab:
        rng = np.random.RandomState(seed=args.rng_seed)
    else:
        rng = None

    val_dat = SingleTimestep(
        args.val_data_path, args.vocab_size, rng=rng, dense=True)

    beam_log_s = np.load(args.log_s_path)[:, :, -1]
    beam_p_s = np.exp(beam_log_s)
    beam_weight = beam_p_s / np.sum(beam_p_s, axis=0)

    reader = tf.train.NewCheckpointReader(args.context_vector_ckpt)
    v = reader.get_tensor('q/mean_v') # shape (V, d)
    beam_mu = np.load(args.mu_path)[:, :, :, -1]

    # log p(n|u) is approximated by its lower bound
    # log p(n|u) \approx sum_i w_i * log p(n|u, i)
    balancing_constant = None
    log_likelihood_i = np.zeros(args.vocab_size)
    log_likelihood_totalnorm_mostlikely = None
    for k, (u, w) in enumerate(zip(beam_mu, beam_weight)):
        log_likelihood_i += val_model.pred_log_likelihood(val_dat, u, v, w)
        if k == 0:
            # de-weight
            log_likelihood_totalnorm_mostlikely = log_likelihood_i / w
    counts_i = np.sum(val_dat.n_sum, axis=1)
    ind = np.where(counts_i != 0)[0]
    # separate: 1/|n| * log p(n|u)
    ave_log_likelihood_i = log_likelihood_i[ind] / counts_i[ind]
    log_likelihood = np.average(ave_log_likelihood_i)
    # total: 1/|n_all| * log p(n_all|u)
    log_likelihood_totalnorm = np.sum(log_likelihood_i) / np.sum(val_dat.n_sum)
    # most likely: 
    log_likelihood_totalnorm_mostlikely = np.sum(log_likelihood_totalnorm_mostlikely) / np.sum(val_dat.n_sum)
    
    # `log_likelihood`, `log_likelihood_totalnorm`,
    # `log_likelihood_totalnorm_mostlikely`
    out_file.write(
        '%f, %f, %f\n' % (log_likelihood, 
                          log_likelihood_totalnorm, 
                          log_likelihood_totalnorm_mostlikely))
    out_file.flush()

