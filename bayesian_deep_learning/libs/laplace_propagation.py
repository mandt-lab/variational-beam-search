import os
import sys
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf

from util import print_log, save_weights, load_weights
from util import LaplacePriorRegularizer
from model import ImageClassificationModel
from distribution_shift_generator import LongTransformedCifar10Generator

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_neural_network_with_regularizer(
        star_vars=[0., 0., 0., 0., 0.], 
        F_total_cov=[1., 1., 1., 1., 1.]):
    neural_net = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_regularizer=LaplacePriorRegularizer(
                1.0, star_vars[0], F_total_cov[0])),
        tf.keras.layers.Conv2D(
            32, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_regularizer=LaplacePriorRegularizer(
                1.0, star_vars[1], F_total_cov[1])),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            64, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_regularizer=LaplacePriorRegularizer(
                1.0, star_vars[2], F_total_cov[2])),
        tf.keras.layers.Conv2D(
            64, kernel_size=3, padding="SAME", activation=tf.nn.relu,
            kernel_regularizer=LaplacePriorRegularizer(
                1.0, star_vars[3], F_total_cov[3])),
        tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            10,
            kernel_regularizer=LaplacePriorRegularizer(
                1.0, star_vars[4], F_total_cov[4]))])
    return neural_net

class CNN(ImageClassificationModel):
    '''This is specific to CIFAR10 dataset. 
    The network architecture is borrowed from 
    "https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-
                                            for-cifar-10-photo-classification/"
    '''
    def __init__(self, x_train, y_train, 
            neural_net, F_total_cov=[1., 1., 1., 1., 1.],
            mc_sample=None, learning_rate=0.001, lam=1.0, rng=None):
        '''Note that `mc_sample` does not work. 
        '''
        del mc_sample # does not work
        self.F_total_cov = F_total_cov
        super().__init__(x_train, y_train, neural_net, 
                         learning_rate=learning_rate,
                         beta=lam,
                         rng=rng)
        
    def get_weights(self):
        names = [layer.name
            for layer in self.neural_net.layers
            if ('conv' in layer.name or 'dense' in layer.name)]
        star_vars = [layer.kernel
            for layer in self.neural_net.layers
            if ('conv' in layer.name or 'dense' in layer.name)]
        self.var_list = star_vars
        star_vars = self.sess.run(star_vars)
        F_total_cov = self.compute_fisher()
        return star_vars, F_total_cov, names

    def compute_fisher(self, num_samples=1000):
        '''computer Fisher information for each parameter.

        `self.var_list` is initilized in `self.get_weights()`.
        '''
        if self.training_size < num_samples:
            num_samples = self.training_size

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(
                np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        # increase numerical stability:
        #   logits - logsumexp(logits)
        self.log_probs = tf.compat.v1.nn.log_softmax(self.logits) 
        # `tf.multinomial` returns shape of (num_data, num_samples)
        self.class_ind = tf.squeeze(
            tf.cast(tf.multinomial(self.logits, 1), tf.int32))

        # select random input image
        im_ind = np.random.randint(self.training_size, size=num_samples)
        # compute first-order derivatives
        print('Computing gradients......')
        self.sample_logprobs = tf.gather_nd(
            self.log_probs, 
            tf.transpose([list(range(num_samples)), self.class_ind]))
        self.ders = tf.gradients(self.sample_logprobs, self.var_list)
        sample_probs, ders = self.sess.run([
            self.sample_logprobs, self.ders], 
            feed_dict={self.images: self.x_train[im_ind]})

        # square the derivatives and divide by number of samples
        print('Computing Fisher information and updating F......')
        for v in range(len(self.F_accum)):
            self.F_accum[v] = (np.square(ders[v]) / num_samples)\
                 * self.training_size
            self.F_total_cov[v] += self.F_accum[v]

        return self.F_total_cov

def laplace_propagation(
        datagen,
        rng,
        save_path,
        max_iter,
        initial_prior_var,
        lr,
        lam,
        independent=False,
        get_neural_net_with_prior=get_neural_network_with_regularizer,
        param_layers_at_most=100,
        surrogate_initial_prior_path=None):
    """
    Args:
    param_layers_at_most -- The number of parameterized layers at most.

    Returns:
    The path of test accuracies.
    """
    test_accs = []
    for task_id in range(max_iter):
        x_train, y_train, x_test, y_test = datagen.next_task()

        # if not debug:
        #     if os.path.exists(save_path + 'weights%d.pkl' % (task_id)):
        #         continue

        if task_id == 0 or independent:
            if surrogate_initial_prior_path is None:
                print("USING INITIAL PRIOR AND "
                      "THE INITIAL PRIOR SUPPORT AT MOST "
                      f"{param_layers_at_most}-LAYER NETWORK.")
                star_vars = [0.] * param_layers_at_most
                F_total_cov = [1./initial_prior_var] * param_layers_at_most
            else:
                print("USE SURROGATE PRIOR DISTRIBUTION.")
                star_vars, F_total_cov = load_weights(
                infile_name=surrogate_initial_prior_path)
        else:
            star_vars, F_total_cov = load_weights(
                infile_name=save_path + 'weights%d.pkl' % (task_id - 1))

        neural_net = get_neural_net_with_prior(star_vars, F_total_cov)

        model = CNN(
            x_train, y_train, 
            neural_net, F_total_cov=F_total_cov,
            mc_sample=None, learning_rate=lr, lam=lam, rng=rng)
        model.init_session()

        model.neural_net.summary()
        
        print_log("Task %d begins" % task_id)
        # train
        (costs, lik_costs, 
         training_accs, val_accs, 
         trainig_ces, val_ces) = model.train(
                batch_size=64, no_epochs=150, display_epoch=10, 
                x_val=x_test, y_val=y_test, verbose=False, test_mc_num=1)
        np.save(save_path + 'train_info_64filter_64batch_%d.npy' % task_id, 
            [costs, lik_costs, 
             training_accs, val_accs, 
             trainig_ces, val_ces])

        star_vars, F_total_cov, var_names = model.get_weights()
        save_weights([star_vars, F_total_cov], 
            outfile_name=save_path + 'weights%d.pkl' % (task_id))

        # test
        test_acc, ave_probs, ave_cross_entropy = model.test_accuracy(
            x_test, y_test)
        print_log(test_acc, ave_cross_entropy)
        print_log("Task %d ends" % task_id)

        test_accs.append(test_acc)
        np.save(save_path + 'test_accs.npy', test_accs)

        tf.reset_default_graph()
    return save_path + 'test_accs.npy'

if __name__ == '__main__':

    tf.reset_default_graph()
    random_seed = 1
    rng = np.random.RandomState(seed=random_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))

    debug = False

    max_iter = 100
    changerate = 3
    task_size = 20000
    lam = 1.0
    lr = 0.001

    folder_name = './lp_res_100epoch_smallpriorvar1e-2/'
    if not debug:
        sys.stdout = open(folder_name + 'log.txt', 'a') # log file

    datagen = LongTransformedCifar10Generator(
            rng=rng, changerate=changerate, 
            max_iter=max_iter, task_size=task_size)

    laplace_propagation(
        datagen=datagen,
        rng=rng,
        save_path=folder_name,
        max_iter=max_iter,
        initial_prior_var=1e-2,
        lr=lr,
        lam=lam,
        get_neural_net_with_prior=get_neural_network_with_regularizer,
        param_layers_at_most=5)
