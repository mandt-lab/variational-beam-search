import os
import sys
from datetime import datetime
import pickle
import abc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from util import print_log, save_weights, load_weights
from util import ind_multivariate_normal_fn
from distribution_shift_generator import LongTransformedCifar10Generator
from distribution_shift_generator import LongTransformedSvhnGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ImageClassificationModel(abc.ABC):
    """Tensorflow model class with a prior over weights.

    The model is compatible with point estimation (MAP) and Bayesian 
    inference (stochastic variational inference). Both paramdigms share the
    same loss function structure, i.e., log_posterior = log_likelihood +
    log_prior. Thus they can use the same code architecture.

    The optimization algorithm is ADAM.
    """
    def __init__(self, x_train, y_train, neural_net, 
                 learning_rate=0.001, beta=1.0, rng=None):
        """`beta` controls the regularization strength.
        """
        self.x_train = x_train
        self.y_train = y_train
        assert self.x_train.shape[0] == self.y_train.shape[0]
        self.training_size = self.x_train.shape[0]

        self.learning_rate = learning_rate
        self.beta = beta
        if rng is None:
            self.rng = np.random.RandomState(2**31)
        else:
            self.rng = rng

        with tf.compat.v1.variable_scope('placeholders'):
            self.images = tf.placeholder(
                tf.float32, 
                shape=[None,
                       self.x_train.shape[1],
                       self.x_train.shape[2],
                       self.x_train.shape[3]])
            self.labels = tf.placeholder(tf.int32, shape=[None,])

        self.neural_net = neural_net

        self.logits = self.neural_net(self.images)
        self.prediction_prob = tf.nn.softmax(self.logits)

        # training
        with tf.compat.v1.variable_scope('training_loss'):
            self.neg_log_likelihood = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.logits))
            # need to divide the total number of training samples to get the
            # correct scale of the ELBO: 
            # notice that we use `reduce_mean` as above
            self.kl = sum(self.neural_net.losses) / self.training_size
            self.nelbo = self.neg_log_likelihood + self.beta * self.kl
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.nelbo)

        # testing
        self.predictions = tf.argmax(self.logits, axis=1)

    # initialization
    def init_session(self):
        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(init)

    def test_accuracy(self, x_test, y_test, mc_num=10):
        """The same test data will be evaluated `mc_num` times.

        Args:
        x_test -- numpy array test images of the same shape as x_train
        y_test -- numpy array test image labels of the same shape as y_train
        mc_num -- number of times for which the test dataset will be evaluated.

        Returns:
        Average test accuracy of all images;
        Probabilities of each class for each test image;
        Cross entropy of the test dataset.
        """
        ave_probs = 0.
        ave_cross_entropy = 0.
        for i in range(mc_num):
            (nlog_lik, prediction_prob) = self.sess.run(
                [self.neg_log_likelihood, self.prediction_prob], 
                feed_dict={
                            self.images: x_test, 
                            self.labels: y_test
                })
            ave_probs += (prediction_prob/mc_num)
            ave_cross_entropy += (nlog_lik/mc_num)
        test_pred = np.argmax(ave_probs, axis=1)
        test_acc = np.sum((test_pred - y_test) == 0)/y_test.shape[0]
        return test_acc, ave_probs, ave_cross_entropy

    def get_elbo(self, batch_size=64, mc_num=10000):
        elbo = 0.
        for i in range(mc_num):
            random_batch_ind = self.rng.randint(
                low=0, 
                high=self.training_size, 
                size=batch_size)
            batch_x_train = self.x_train[random_batch_ind, ...]
            batch_y_train = self.y_train[random_batch_ind]
            _elbo = -self.sess.run(
                self.nelbo,
                feed_dict={
                    self.images: batch_x_train, 
                    self.labels: batch_y_train
                })
            elbo += _elbo
        elbo /= mc_num
        return elbo

    # Train model
    def train(self, batch_size=200, no_epochs=1000, display_epoch=100, 
            x_val=None, y_val=None, verbose=False, test_mc_num=10):
        
        num_batch = int(np.ceil(1.0 * self.training_size / batch_size))

        sess = self.sess
        costs = []
        lik_costs = []
        training_accs = []
        val_accs = []
        # only an estimation because it does not consider the randomness in `y`
        trainig_ces = []
        val_ces = [] 

        # Training cycle
        for epoch in range(no_epochs):

            epoch_cost = 0.
            epoch_lik_cost = 0.

            random_ind = list(range(self.training_size)) 
            self.rng.shuffle(random_ind)
            for batch_id in range(num_batch):
                batch_start = batch_id*batch_size
                batch_end = (batch_id+1)*batch_size
                random_batch_ind = random_ind[batch_start:batch_end]
                batch_x_train = self.x_train[random_batch_ind, ...]
                batch_y_train = self.y_train[random_batch_ind]
                # print_log(batch_x_train.shape, batch_y_train.shape)

                _, c_total, lik_loss, _, _ = sess.run(
                    [self.train_step, self.nelbo, self.neg_log_likelihood,
                     self.predictions, self.labels],
                    feed_dict={
                        self.images: batch_x_train, 
                        self.labels: batch_y_train
                    })

                # Compute average loss
                epoch_cost += (c_total/num_batch)
                epoch_lik_cost += (lik_loss/num_batch)

            # Display logs every display_epoch
            if epoch == 0 \
                    or (epoch+1) % display_epoch == 0 \
                    or epoch == no_epochs-1:
                print_log("Epoch:", '%04d' % (epoch + 1), 
                    "total cost=", 
                    "{:.9f}".format(epoch_cost), 
                    "log-likelihood term=", 
                    "{:.9f}".format(epoch_lik_cost), 
                    "kl/regluarization term=", 
                    "{:.9f}".format(epoch_cost - epoch_lik_cost))
                if verbose:
                    train_acc, _, train_cross_entropy = self.test_accuracy(
                        self.x_train[:10000], self.y_train[:10000], test_mc_num)
                    print_log(
                        "\ttraining accuracy=", "{:.5f}".format(train_acc),
                        "training crosss entropy=", 
                            "{:.5f}".format(train_cross_entropy))
                    if x_val is not None and y_val is not None:
                        val_acc, _, val_cross_entropy = self.test_accuracy(
                            x_val, y_val, test_mc_num)
                        print_log(
                            "\tvalidation accuracy=", "{:.5f}".format(val_acc),
                            "validation crosss entropy=", 
                                "{:.5f}".format(val_cross_entropy))
                        val_accs.append(val_acc)
                        val_ces.append(val_cross_entropy)
                    training_accs.append(train_acc)
                    trainig_ces.append(train_cross_entropy)
                    
            costs.append(epoch_cost)
            lik_costs.append(epoch_lik_cost)

        print_log("Optimization Finished!")

        return costs, lik_costs, training_accs, val_accs, trainig_ces, val_ces


def get_bayesian_neural_net_with_prior(
        prior_m=[None, None, None, None, None, None, None], 
        prior_s=[None, None, None, None, None, None, None],
        initial_prior_var=1.):
    """
    Args:
    prior_m -- A list of prior mean
    prior_s -- A list of prior standard deviation
    initial_prior_var -- Initial prior variance is set if either prior_m or 
        prior_s is None.

    Returns:
    A tensorflow keras Sequential object as a model architecture.
    """
    kwargs = {
        'kernel_posterior_tensor_fn': (
            lambda d: d.sample()),
    }
    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DFlipout(
            32, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=tf.initializers.random_normal(stddev=0.1),
                untransformed_scale_initializer=tf.initializers.random_normal(
                    mean=-3.0, stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
                prior_var=initial_prior_var, mu=prior_m[0], sigma=prior_s[0]), 
            **kwargs),
        tfp.layers.Convolution2DFlipout(
            32, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=tf.initializers.random_normal(stddev=0.1),
                untransformed_scale_initializer=tf.initializers.random_normal(
                    mean=-3.0, stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
                prior_var=initial_prior_var, mu=prior_m[1], sigma=prior_s[1]), 
            **kwargs),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Dropout(0.2),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=tf.initializers.random_normal(stddev=0.1),
                untransformed_scale_initializer=tf.initializers.random_normal(
                    mean=-3.0, stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
                prior_var=initial_prior_var, mu=prior_m[2], sigma=prior_s[2]), 
            **kwargs),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=tf.initializers.random_normal(stddev=0.1),
                untransformed_scale_initializer=tf.initializers.random_normal(
                    mean=-3.0, stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
                prior_var=initial_prior_var, mu=prior_m[3], sigma=prior_s[3]), 
            **kwargs),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(10,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                loc_initializer=tf.initializers.random_normal(stddev=0.1),
                untransformed_scale_initializer=tf.initializers.random_normal(
                    mean=-3.0, stddev=0.1)),
            kernel_prior_fn=ind_multivariate_normal_fn(
                prior_var=initial_prior_var, mu=prior_m[4], sigma=prior_s[4]), 
            **kwargs)])
    return neural_net


class BayesianCNN(ImageClassificationModel):
    '''This is specific to CIFAR10 dataset. 
    The network architecture is modified from Section A in
    "http://proceedings.mlr.press/v70/zenke17a/zenke17a-supp.pdf"
    and "https://machinelearningmastery.com/how-to-develop-a-cnn-from-
                                scratch-for-cifar-10-photo-classification/"

    The code is borrowed from:
    "https://medium.com/python-experiments/bayesian-cnn-model-on-mnist-data-
                    using-tensorflow-probability-compared-to-cnn-82d56a298f45"
    "https://github.com/tensorflow/probability/blob/master/
                    tensorflow_probability/examples/bayesian_neural_network.py"
    '''
    def __init__(self, x_train, y_train, neural_net,
            mc_sample=None, learning_rate=0.001, beta=1.0, rng=None):
        '''Note that `mc_sample` does not work. 
        '''
        del mc_sample # does not work
        super().__init__(x_train, y_train, neural_net, learning_rate, beta, rng)

    def get_weights(self):
        names = [layer.name for layer in self.neural_net.layers
            if 'flipout' in layer.name]
        qm = [layer.kernel_posterior.mean()
            for layer in self.neural_net.layers
            if 'flipout' in layer.name]
        qs = [layer.kernel_posterior.stddev()
            for layer in self.neural_net.layers
            if 'flipout' in layer.name]
        qm_vals, qs_vals = self.sess.run([qm, qs])
        return qm_vals, qs_vals, names

def vcl(datagen,
        rng,
        save_path,
        max_iter,
        initial_prior_var,
        beta,
        lr,
        independent,
        get_neural_net_with_prior=get_bayesian_neural_net_with_prior,
        param_layers_at_most=100,
        surrogate_initial_prior_path=None):
    if os.path.exists(save_path + 'test_accs.npy'):
        test_accs = list(
            np.load(save_path + 'test_accs.npy', allow_pickle=True))
    else:
        test_accs = []

    for task_id in range(max_iter):
        x_train, y_train, x_test, y_test = datagen.next_task()

        if os.path.exists(save_path + 'weights%d.pkl' % (task_id)):
            continue

        if task_id == 0 or independent:
            if surrogate_initial_prior_path is None:
                print("THE INITIAL PRIOR SUPPORT AT MOST "
                      f"{param_layers_at_most}-LAYER NETWORK.")
                qm_vals = [None] * param_layers_at_most
                qs_vals = [None] * param_layers_at_most
            else:
                print("USE SURROGATE PRIOR DISTRIBUTION.")
                qm_vals, qs_vals = load_weights(
                infile_name=surrogate_initial_prior_path)
        else:
            qm_vals, qs_vals = load_weights(
                infile_name=save_path + 'weights%d.pkl' % (task_id - 1))

        neural_net = get_neural_net_with_prior(qm_vals, 
                                               qs_vals,
                                               initial_prior_var)

        model = BayesianCNN(
            x_train, y_train, neural_net,
            mc_sample=None, learning_rate=lr, beta=beta, rng=rng) # 0.001
        model.init_session()
        model.neural_net.summary()
        
        print_log("Task %d begins" % task_id)
        # train
        (costs, lik_costs, 
         training_accs, val_accs, 
         trainig_ces, val_ces) = model.train(
                batch_size=64, no_epochs=150, display_epoch=10, 
                x_val=x_test, y_val=y_test, verbose=False)
        np.save(save_path + 'train_info_64filter_64batch_%d.npy' % task_id, 
            [costs, lik_costs, 
             training_accs, val_accs, 
             trainig_ces, val_ces])

        qm_vals, qs_vals, q_names = model.get_weights()
        save_weights([qm_vals, qs_vals], 
            outfile_name=save_path + 'weights%d.pkl' % (task_id))

        # print_log("Getting ELBO:")
        # elbo = model.get_elbo()
        # print_log(f"\t{elbo}")

        # test
        test_acc, ave_probs, ave_cross_entropy = model.test_accuracy(x_test, y_test)
        print_log("Accuracy and cross entropy:", test_acc, ave_cross_entropy)
        print_log("Task %d ends" % task_id)

        test_accs.append(test_acc)
        np.save(save_path + 'test_accs.npy', test_accs)

    return save_path + 'test_accs.npy'



def bayesian_broadening(qm_vals, qs_vals, pm_vals, ps_vals, scale_diffusion):
    _qm_vals, _qs_vals = [], []
    for qm, qs, pm, ps in zip(qm_vals, qs_vals, pm_vals, ps_vals):
        s_0 = ps / (1-scale_diffusion)
        s_1 = qs / scale_diffusion
        v_0 = s_0**2
        v_1 = s_1**2
        # take product
        qs = s_0 * s_1 / np.sqrt(v_0 + v_1)
        qm = (pm*v_1 + qm*v_0) / (v_0 + v_1)

        _qm_vals.append(qm)
        _qs_vals.append(qs)
    return _qm_vals, _qs_vals

def bf(datagen,
        rng,
        save_path,
        max_iter,
        initial_prior_var,
        beta,
        diffusion,
        lr,
        independent,
        get_neural_net_with_prior=get_bayesian_neural_net_with_prior,
        param_layers_at_most=100,
        surrogate_initial_prior_path=None):
    if os.path.exists(save_path + 'test_accs.npy'):
        test_accs = list(
            np.load(save_path + 'test_accs.npy', allow_pickle=True))
    else:
        test_accs = []

    for task_id in range(max_iter):
        x_train, y_train, x_test, y_test = datagen.next_task()

        if os.path.exists(save_path + 'weights%d.pkl' % (task_id)):
            continue

        if task_id == 0 or independent:
            if surrogate_initial_prior_path is None:
                print("THE INITIAL PRIOR SUPPORT AT MOST "
                      f"{param_layers_at_most}-LAYER NETWORK.")
                qm_vals = [None] * param_layers_at_most
                qs_vals = [None] * param_layers_at_most
            else:
                print("USE SURROGATE PRIOR DISTRIBUTION.")
                qm_vals, qs_vals = load_weights(
                infile_name=surrogate_initial_prior_path)
        else:
            print("Perform Bayesian Forgetting")
            scale_diffusion = np.sqrt(diffusion)
            if surrogate_initial_prior_path is None:
                print("\tUse standard Gaussian as prior distribution.")
                pm_vals = [0.] * param_layers_at_most
                ps_vals = [1.] * param_layers_at_most
            else:
                print("\tUse surrogate prior distribution.")
                pm_vals, ps_vals = load_weights(
                infile_name=surrogate_initial_prior_path)

            qm_vals, qs_vals = load_weights(
                infile_name=save_path + 'weights%d.pkl' % (task_id - 1))

            # broadening
            qm_vals, qs_vals = bayesian_broadening(qm_vals, qs_vals, pm_vals, ps_vals, scale_diffusion)

        neural_net = get_neural_net_with_prior(qm_vals, 
                                               qs_vals,
                                               initial_prior_var)

        model = BayesianCNN(
            x_train, y_train, neural_net,
            mc_sample=None, learning_rate=lr, beta=beta, rng=rng) # 0.001
        model.init_session()
        model.neural_net.summary()
        
        print_log("Task %d begins" % task_id)
        # train
        (costs, lik_costs, 
         training_accs, val_accs, 
         trainig_ces, val_ces) = model.train(
                batch_size=64, no_epochs=150, display_epoch=10, 
                x_val=x_test, y_val=y_test, verbose=False)
        np.save(save_path + 'train_info_64filter_64batch_%d.npy' % task_id, 
            [costs, lik_costs, 
             training_accs, val_accs, 
             trainig_ces, val_ces])

        qm_vals, qs_vals, q_names = model.get_weights()
        save_weights([qm_vals, qs_vals], 
            outfile_name=save_path + 'weights%d.pkl' % (task_id))

        # print_log("Getting ELBO:")
        # elbo = model.get_elbo()
        # print_log(f"\t{elbo}")

        # test
        test_acc, ave_probs, ave_cross_entropy = model.test_accuracy(x_test, y_test)
        print_log("Accuracy and cross entropy:", test_acc, ave_cross_entropy)
        print_log("Task %d ends" % task_id)

        test_accs.append(test_acc)
        np.save(save_path + 'test_accs.npy', test_accs)

    return save_path + 'test_accs.npy'

if __name__ == '__main__':

    tf.reset_default_graph()
    random_seed = 1
    rng = np.random.RandomState(seed=random_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))
    rng_for_model = np.random.RandomState(seed=random_seed)

    max_iter = 100
    changerate = 3
    task_size = 20000
    independent = False

    initial_prior_var = 1.0 
    beta = 1.0
    lr = 0.0001

    folder_name = './test_case_vcl/'
    sys.stdout = open(folder_name + 'log.txt', 'w')

    datagen = LongTransformedSvhnGenerator(
            rng=rng, changerate=changerate, 
            max_iter=max_iter, task_size=task_size)

    vcl(datagen=datagen,
        rng=rng_for_model,
        save_path=folder_name,
        max_iter=max_iter,
        initial_prior_var=initial_prior_var,
        beta=beta,
        lr=lr,
        independent=independent,
        get_neural_net_with_prior=get_bayesian_neural_net_with_prior,
        param_layers_at_most=100)
