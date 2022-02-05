'''Bayesian word embedding model for both static fits and dynamic filtering.

Basic usage:

    python3 single-timestep.py [optional arguments] input_file output_directory

where `input_file` contains a sparse word-context co-occurrence matrix. Use the
script TODO to generate such a matrix from a corpus. For more detailed
description, run: `python3 single-timestep.py -h`.

This script can be used for two different tasks:

a) fitting a Bayesian word embedding model to a single corpus; or
b) performing a single time step of the Skip-Gram filtering algorithm proposed
            Mandt, "Dynamic Word Embeddings", ICML 2017].

To do Skip-Gram filtering, run this script sequentially for each time step.
From the second time step on, provide the command line argument
`--previous_timestep_mu/std` and point it to the output directory of the previous
fit. You can control the diffusion constant with `--diffusion`.

TODO: Provide a shell script that runs the filtering loop.
'''

from lib import training, optimizer, dataset

import os
import imp
import datetime
import argparse
import sys
import struct
import gzip
import numpy as np
import tensorflow as tf
tfd = tf.distributions

# mute tensorflow information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}


class Model(training.Model):
    '''A Bayesian Skip-Gram model, similar to the model proposed            )].

    Different to the proposal           )], we fit the model using black
    box variational inference. This model is currently mainly intended for toy
    experiments. A future implementation of the dynamic filtering algorithm
    [Bamler & Mandt (2017)] will build on this model.

    [Barkan (2016)]: https://arxiv.org/abs/1603.06571
    [Bamler & Mandt (2017)]: http://proceedings.mlr.press/v70/bamler17a.html
    '''

    def __init__(self, args, rng, log_file):
        '''Create the model.

        Arguments:
        args -- A namespace of command line arguments, as returned by
            `argparse.ArgumentParser.parse_args()`. See also function
            `add_cli_args()` in this module.
        rng -- An `np.random.RandomState`. Will be used to seed the sampler from
            the variational distribution.
        log_file -- A file handle for log messages. Log messages will be written
            in the form of python statements so that the log file can easily be
            parsed by other python scripts.
        '''

        log_file.write('# Loading training data... ')
        log_file.flush()
        if args.rand_vocab:
            dat = dataset.SingleTimestep(
                args.input, args.max_vocab_size, dense=args.minibatch_size is None,
                neg_ratio=args.neg_ratio, neg_exp=args.neg_exponent, rng=rng, 
                norm_npos=args.norm_npos)
        else:
            dat = dataset.SingleTimestep(
                args.input, args.max_vocab_size, dense=args.minibatch_size is None,
                neg_ratio=args.neg_ratio, neg_exp=args.neg_exponent,
                norm_npos=args.norm_npos)
        self.dat = dat
        log_file.write('done.\n')
        log_file.write('vocab_size = %d\n' % dat.vocab_size)
        if dat.time_stamp is None:
            log_file.write('time_stamp = None')
        else:
            log_file.write('time_stamp = "%s"\n' % str(dat.time_stamp))

        if args.minibatch_size is None:
            self.minibatch_size = None
        else:
            assert args.minibatch_size < dat.vocab_size
            self.minibatch_size = args.minibatch_size

        if args.tokenid_detected_file is None:
            self.tokenid_detected = None
        else:
            self.tokenid_detected = np.loadtxt(args.tokenid_detected_file, dtype=np.int32)

        self.s_t = args.s_t

        # Comment out to use in-graph variables because feed_dict is single-threaded and slow.
        # See: https://stackoverflow.com/questions/47298447/how-to-fix-low-volatile-gpu-util-with-tensorflow-gpu-and-keras
        # and: https://stackoverflow.com/questions/35394103/initializing-tensorflow-variable-with-an-array-larger-than-2gb
        # TODO: use input pipeline
        # self.n_sum, self.counts_pos, self.scaled_freq_neg, self.words, self.ctxts = self._get_placeholders(
        #     dat, args)
        with tf.variable_scope('data'):
            self.n_sum_place = tf.placeholder(tf.float32, 
                                    shape=(self.dat.vocab_size, self.dat.vocab_size))
            self.n_sum = tf.Variable(self.n_sum_place, dtype=tf.float32, name='n_sum', trainable=False)
            self.counts_pos = tf.constant(
                    self.dat.counts_pos, dtype=tf.float32, name='counts_pos')
            self.scaled_freq_neg = tf.constant(
                    self.dat.scaled_freq_neg, dtype=tf.float32, name='scaled_freq_neg')
        # self.words = None
        # self.ctxts = None

        prior = self._get_prior_parameters(args, dat, log_file)
        self._define_variational_distribution(args, dat, prior)
        minus_lp = self._minus_expected_log_prior(args, prior)

        if args.map:
            minus_ll = self._negative_log_likelihood(
                self.n_sum, self.counts_pos, self.scaled_freq_neg, self.mean_u, self.mean_v)
        else:
            u = self._sample_from_q(rng)
            minus_ll = self._negative_log_likelihood(
                self.n_sum, self.counts_pos, self.scaled_freq_neg, u, self.mean_v)
            entropy = self._entropy_q()

        if self.minibatch_size is not None:
            minus_ll *= dat.vocab_size / args.minibatch_size

        with tf.variable_scope('loss'):
            if args.map:
                self.loss_components = minus_lp + minus_ll
                self.loss = tf.reduce_sum(self.loss_components)
            else:
                # Define the loss as the estimate of the negative ELBO.
                self.loss_components = minus_lp + minus_ll - args.temper * entropy # shape (V,)
                self.loss = tf.reduce_sum(self.loss_components)

        self._opt_step = optimizer.define_optimizer(
            self.loss, tf.trainable_variables(), args,
            minibatch=self.minibatch_size is not None)

        self.elbo = - self.loss_components # shape (V,)

        tf.summary.scalar('training_loss', self.loss)
        # tf.summary.scalar('npos_sum', tf.reduce_sum(self.counts_pos))
        # tf.summary.histogram('npos', self.counts_pos)
        # tf.summary.scalar('negative_log_likelihood', minus_ll)

    def _get_prior_parameters(self, args, dat, log_file):
        '''Calculate the prior mean and precision for a filtering step.

        If --previous_timestep_mu/std is set then this function loads the last
        checkpoint from the previous time step and propagates the approximate
        posterior forward in time according to an Ornstein Uhlenbeck process.

        If --previous_timestep_mu/std is not set then the prior is a Gaussian with
        standard deviation --prior_std centered around zero.

        Returns:
        A dict holding the prior means and the diagonal elements of (half) the
        prior precision matrix.
        '''
        full_mean_v = self._get_context_vector(args)

        if args.previous_timestep_mu is None and args.previous_timestep_std is None:
            full_mean_u = tf.zeros((dat.vocab_size, args.embedding_dim))
            full_half_precision_u = tf.fill(
                (dat.vocab_size, args.embedding_dim), 0.5 / args.prior_std**2)
        else:
            if dat.time_stamp is not None:
                # Load log file of previous time step to read its time stamp
                previous_date = ... # TODO
                time_diff = (dat.time_stamp - previous_date).days / 365.0
            else:
                time_diff = 1
            log_file.write('years_since_previous_step = %g\n' % time_diff)
            broadening = args.diffusion * time_diff

            beam_mu = np.load(args.previous_timestep_mu)
            beam_std = np.load(args.previous_timestep_std)

            log_file.write(
                '# Loading checkpoint from previous time step and calculating prior... ')
            log_file.flush()

            old_mean = beam_mu[args.beam_id, :, :, -1]
            old_std = beam_std[args.beam_id, :, :, -1]
            
            if args.map:
                full_mean_u, full_half_precision_u = self._propagate_ornstein_uhlenbeck(
                    old_mean, 0.0,
                    broadening, args.prior_std,
                    args.mult_diff)
            else:
                full_mean_u, full_half_precision_u = self._propagate_ornstein_uhlenbeck(
                    old_mean,
                    old_std,
                    broadening, args.prior_std,
                    args.mult_diff)
            log_file.write('done.\n')
            log_file.flush()

        if self.minibatch_size is None:
            new_mean_u = full_mean_u
            new_half_precision_u = full_half_precision_u
            new_mean_v = full_mean_v
        else:
            new_mean_u = tf.gather(full_mean_u, self.words)
            new_half_precision_u = tf.gather(full_half_precision_u, self.words)
            new_mean_v = tf.gather(full_mean_v, self.ctxts)

        return {
            'full_mean_u': full_mean_u,
            'full_half_precision_u': full_half_precision_u,
            'full_mean_v': full_mean_v,
            'mean_u': new_mean_u,
            'half_precision_u': new_half_precision_u,
            'mean_v': new_mean_v,
        }

    def _propagate_ornstein_uhlenbeck(self, old_mean, old_std, broadening, prior_std, mult_diff):
        '''Implements Eq. 13 of Bamler & Mandt, "Dynamic Word Embeddings", ICML 2017.

        Takes the approximate posterior of the previous time step (`old_mean`
        and `old_std`) and propagates it forward in time. The parameter
        `broadening` is defined in Eq. 3 of the above mentioned paper.

        When `broadening` is set to 0, it is equivalent to ``spike'' kernel and 
        keeps the prior the same as last step's posterior.

        All these computation can be numpy computation.
        '''
        if self.tokenid_detected is None:
            # jump variable is global and controls the whole vocabulary
            if self.s_t == 0:
                new_mean = old_mean
                new_half_precision = 0.5 / old_std**2
            else:
                if mult_diff:
                    broadened_variance = (old_std**2) * broadening
                else:
                    broadened_variance = old_std**2 + broadening
                new_mean = (prior_std**2 * old_mean /
                            (prior_std**2 + broadened_variance))
                new_half_precision = 0.5 / broadened_variance + 0.5 / prior_std**2
        else:
            if mult_diff:
                broadened_variance = (old_std**2) * broadening
            else:
                broadened_variance = old_std**2 + broadening
            new_mean = (prior_std**2 * old_mean /
                        (prior_std**2 + broadened_variance))
            new_half_precision = 0.5 / broadened_variance + 0.5 / prior_std**2
            if self.s_t == 0:
                new_mean[self.tokenid_detected, :] = old_mean[self.tokenid_detected, :]
                new_half_precision[self.tokenid_detected, :] = 0.5 / old_std[self.tokenid_detected, :]**2
        return new_mean, new_half_precision

    def _get_context_vector(self, args):
        assert args.context_vector_ckpt is not None
        reader = tf.train.NewCheckpointReader(args.context_vector_ckpt)
        v = reader.get_tensor('q/mean_v')
        return v

    def _define_variational_distribution(self, args, dat, prior):
        with tf.variable_scope('q'):
            if args.map:
                self.full_mean_u = tf.get_variable('mean_u', shape=[dat.vocab_size, args.embedding_dim],
                                                   initializer=tf.glorot_uniform_initializer())
                self.full_mean_v = tf.Variable(
                    prior['full_mean_v'], dtype=tf.float32, name='mean_v', trainable=False)

                if self.minibatch_size is None:
                    self.mean_u = self.full_mean_u
                    self.mean_v = self.full_mean_v
                else:
                    self.mean_u = tf.gather(self.full_mean_u, self.words)
                    self.mean_v = tf.gather(self.full_mean_v, self.ctxts)
            else:
                self.full_mean_u = tf.Variable(
                    prior['full_mean_u'], dtype=tf.float32, name='mean_u')
                self.full_log_std_u = tf.Variable(
                    -0.5 * tf.log(2 * prior['full_half_precision_u']), dtype=tf.float32, name='log_std_u')

                self.full_mean_v = tf.Variable(
                    prior['full_mean_v'], dtype=tf.float32, name='mean_v', trainable=False)

                if self.minibatch_size is None:
                    self.mean_u = self.full_mean_u
                    self.mean_v = self.full_mean_v
                    self.log_std_u = self.full_log_std_u
                else:
                    self.mean_u = tf.gather(self.full_mean_u, self.words)
                    self.mean_v = tf.gather(self.full_mean_v, self.ctxts)
                    self.log_std_u = tf.gather(self.full_log_std_u, self.words)

                self.std_u = tf.exp(self.log_std_u, name='std_u')
                self.q_u = tfd.Normal(
                    loc=self.mean_u, scale=self.std_u, name='q_u')

            tf.summary.histogram('mean_u', self.mean_u)
            tf.summary.histogram('mean_v', self.mean_v)
            tf.summary.histogram('motion_u', self.mean_u - prior['mean_u'])
            tf.summary.histogram('std_u', self.std_u)

    def _minus_expected_log_prior(self, args, prior):
        with tf.variable_scope('log_prior'):
            if args.map:
                log_prior_u = tf.reduce_sum(prior['half_precision_u'] *
                                            ((self.mean_u - prior['mean_u'])**2) -
                                            tf.log(prior['half_precision_u'] * 2) / 2,
                                            axis=1)
            else:
                # Extra `log(std)` term is used for elbo evaluation but does not
                # help optimization.
                log_prior_u = tf.reduce_sum(prior['half_precision_u'] *
                                            ((self.mean_u - prior['mean_u'])**2 + self.std_u**2) -
                                            tf.log(prior['half_precision_u'] * 2) / 2,
                                            axis=1)
            return log_prior_u

    def _entropy_q(self):
        with tf.variable_scope('entropy'):
            return tf.reduce_sum(self.log_std_u, axis=1)

    def _sample_from_q(self, rng):
        with tf.variable_scope('sampling'):
            u = self.q_u.sample(seed=rng.randint(0, 2**16))  # shape (V, d)
            return u

    def _get_placeholders(self, dat, args):
        if self.minibatch_size is None:
            minibatch_vocab_size = dat.vocab_size
            words = None
            contexts = None
        else:
            minibatch_vocab_size = args.minibatch_size
            words = tf.placeholder(tf.int32, shape=(minibatch_vocab_size))
            contexts = tf.placeholder(tf.int32, shape=(minibatch_vocab_size))

        n_sum = tf.placeholder(tf.float32, shape=(
            minibatch_vocab_size, minibatch_vocab_size))
        counts_pos = tf.placeholder(tf.float32, shape=(minibatch_vocab_size,))
        scaled_freq_neg = tf.placeholder(
            tf.float32, shape=(minibatch_vocab_size,))
        return n_sum, counts_pos, scaled_freq_neg, words, contexts

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

    @property
    def opt_step(self):
        return self._opt_step

    def init_feed_dict(self):
        return {self.n_sum_place: self.dat.n_sum}

    # Comment out to use in-graph variables because feed_dict is 
    # single-threaded and slow.
    # TODO: use input pipeline
    # def generate_feed_dict(self, rng):
    #     '''Construct feed_dict for sess.run().

    #     Arguments:
    #     dat -- Dataset handler obtained from class Dataset
    #     n_sum -- Placeholder for n_sum
    #     counts_pos -- Placeholder for counts of words
    #     scaled_freq_neg -- Placeholder for scaled_freq_neg

    #     Returns:
    #     A dict for feed_dict
    #     '''

    #     if self.minibatch_size is None:
    #         return {
    #             self.n_sum: self.dat.n_sum,
    #             self.counts_pos: self.dat.counts_pos,
    #             self.scaled_freq_neg: self.dat.scaled_freq_neg
    #         }
    #     else:
    #         words = rng.choice(
    #             self.dat.vocab_size, self.minibatch_size, replace=False).astype(np.int32)
    #         ctxts = rng.choice(
    #             self.dat.vocab_size, self.minibatch_size, replace=False).astype(np.int32)
    #         # TODO: sort words and ctxts to improve memory locality
    #         n_pos, counts_pos_mb, scaled_freq_neg_mb = self.dat.minibatch(
    #             words, ctxts)

    #         # TODO: calculate `n_sum` in Tensorflow rather than in numpy
    #         n_sum_mb = (
    #             n_pos + counts_pos_mb[:, np.newaxis] * scaled_freq_neg_mb[np.newaxis, :])

    #         return {
    #             self.n_sum: n_sum_mb,
    #             self.counts_pos: counts_pos_mb,
    #             self.scaled_freq_neg: scaled_freq_neg_mb,
    #             self.words: words,
    #             self.ctxts: ctxts
    #         }

    def post_training_action(self, session, step, rng, log_file):
        elbo_mc_size = 10
        elbo = 0.0
        for i in range(elbo_mc_size):
            elbo += session.run([self.elbo], feed_dict=self.generate_feed_dict(rng))[0]
        elbo /= elbo_mc_size

        # tempering elbo # MIGHT AMPLIFY THE NOISE IN CHANGE VARIABLE POSTERIOR!!!!
        # It is even severe for rare words. The detected changes might be quite noisy for rare words.
        # It is not a problem in supervised learning models because the dat_sum is always the same for all timesteps.
        dat_sum = np.sum(self.dat.n_sum, axis=1) # shape (V,)
        ind = np.where(dat_sum == 0)[0]
        dat_sum[ind] = 1
        # elbo /= dat_sum
        
        elbo_file = os.path.dirname(log_file.name) + '/elbo.npy'
        np.save(elbo_file, elbo)
        dat_sum_file = os.path.dirname(log_file.name) + '/dat_sum.npy'
        np.save(dat_sum_file, dat_sum)

def add_cli_args(parser):
    '''Add command line arguments specific to a Bayesian word embedding model.

    This function defines command line arguments that are relevant for a
    Bayesian word embedding model. The constructor of `Model` in this module
    expects an `args` namespace that contains the parameters defined in this
    function.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was added to the parser.
    '''

    group = parser.add_argument_group('Model Parameters')
    group.add_argument('-d', '--embedding_dim', metavar='N', type=int, default=100, help='''
        Set the embedding dimension.''')
    group.add_argument('-B', '--minibatch_size', metavar='N', type=int, help='''
        Size of the minibatch (number of randomly drawn words and randomly drawn contexts per
        training step). If not provided then training will occur on full batches''')
    group.add_argument('-V', '--max_vocab_size', metavar='N', type=int, default=-1, help='''
        Set the maximum vocabulary size. A negative value(default) means that
        the vocabulary size is read from the input data.''')
    group.add_argument('--norm_npos', action='store_true', help='''
        Normalize the `npos` matrix with the total number of words.''')
    group.add_argument('--rand_vocab', action='store_true', help='''
        Randomly select `max_vocab_size` words as vocabulary instead of top `max_vocab_size` words.''')
    group.add_argument('--tokenid_detected_file', metavar='FILE_PATH', help='''
        A set of words that the jump variable accounts for. Each word is represented by its index
        in the vocabulary. Indices are separated by whitespace. If not provided, the jump
        variable accounts for the whole vocabulary.''')
    group.add_argument('--context_vector_ckpt', metavar='FILE_PATH', help='''
        Path to fixed context vectors `V` in the format of TensorFlow checkpoint. The name
        follows `q/mean_v`''')
    group.add_argument('--beam_id', metavar='INT', type=int, default=0, help='''
        Id of current beam. Values in [0, `beam_size`).''')
    group.add_argument('--s_t', metavar='INT', type=int, default=1, help='''
        Value of jump variable for current time step.''')
    group.add_argument('--prior_std', metavar='FLOAT', type=float, default=1.0, help='''
        Standard deviation of the Gaussian prior. If `--previous_timestep_mu/std` is set, then
        `--prior_std` specifies the standard deviation of the additional prior around zero, see
        Eq. 4 in Bamler & Mandt, "Dynamic Word Embeddings", ICML 2017.''')
    group.add_argument('--temper', metavar='FLOAT', type=float, default=1.0, help='''
        The weight of entropy in variational inference. Large weight encourages large 
        variational variance.''')
    group.add_argument('--map', action='store_true', help='''
        Perform maximum a posterior estimation instead of inferring posterior distribution.''')
    group.add_argument('--neg_ratio', metavar='FLOAT', type=float, default=1.0, help='''
        Number of negative samples per positive sample.''')
    group.add_argument('--neg_exponent', metavar='FLOAT', type=float, default=0.75, help='''
        Set the exponent $\\gamma$ for context frequencies in negative samples. Negative samples
        $(i, j)$ are modelled as if words $i$ and contexts $j$ were drawn independently, with
        $p(i)$ being the marginal distribution of words in the corpus, and $p(j)$ being a
        multinomial distribution with probabilities proportional to the frequencies of positive
        samples exponentiated by $\\gamma$. A value of $\\gamma < 1$ makes contexts of negative
        samples more uniformly distributed than those of positive samples.''')
    group.add_argument('--previous_timestep_mu', metavar='DIRECTORY_PATH', help='''
        Path to `beam_mu_{year}.npy` file of shape (beam_size, vocab_size, emb_dim, num_steps).''')
    group.add_argument('--previous_timestep_std', metavar='DIRECTORY_PATH', help='''
        PPath to `beam_std_{year}.npy` file of shape (beam_size, vocab_size, emb_dim, num_steps).''')
    group.add_argument('--diffusion', metavar='FLOAT', type=float, default=0.001, help='''
        Only used if `--previous_timestep_mu/std` is specified. Diffusion constant $D$ as defined in
        Bamler & Mandt, "Dynamic Word Embeddings", ICML 2017. The smaller the diffusion constant,
        the tighter consecutive time steps are coupled to each other. If the data set of the
        current and previous time step contains a time stamp, then the diffusion constant is
        interpreted as diffusion * per year * (more precisely, per 365 days). If the data sets do not
        contain any time stamps then the time difference is implicitly assumed to be one year.''')
    group.add_argument('--mult_diff', action='store_true', help='''
        Use nonlinear multiplicative broadening if set true. Otherwise use traditional additive 
        diffusion constant.''')

    return group


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Fit a single instance of a Bayesian interpretation of the Skip-Gram model('word2vec') to a
        a preprocessed corpus. Expects input data in the form of a sparse word-context
        co-occurrence matrix, as produced by the script TODO.''')
    training.add_cli_args(parser)
    add_cli_args(parser)
    optimizer.add_cli_args(parser)

    training.train(Model, parser)
