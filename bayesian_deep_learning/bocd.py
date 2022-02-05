import os
import sys
from datetime import datetime
import pickle
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

sys.path.extend(['libs/'])
from model import BayesianCNN, get_bayesian_neural_net_with_prior
from util import print_log, save_weights, load_weights
from util import broaden_weights
from distribution_shift_generator import LongTransformedCifar10Generator
from distribution_shift_generator import LongTransformedSvhnGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RunLength:
    def __init__(self, r, prob, params):
        self.r = r
        self.prob = prob
        self.params = params
        self.pred_prob = None # current evidence
        self.factor = None # self.prob*self.pred_prob
        self.test_pred = None # predictions for the next observation
        self.test_acc = None
        self.test_ce = None
        
def prune(run_lens, normalizer, res_num=1):
    run_lens.sort(reverse=True, key=lambda x: x.prob)
    if len(run_lens) <= res_num:
        return run_lens, normalizer
        
    for rl in run_lens[res_num:]:
        normalizer -= rl.prob
    run_lens = run_lens[:res_num]
    return run_lens, normalizer

def _process(
        # model specific
        rl, x_train, y_train, x_test, y_test,
        # environment
        rng, first_gpu, max_gpu,
        # optimization
        initial_prior_var, lr, beta,
        # others
        get_neural_net_with_prior):

    # set visible gpus
    proc = multiprocessing.current_process()
    proc_id = int(proc.name.split('-')[-1])
    # each process assumes one specific gpu
    gpu_id = first_gpu + proc_id % max_gpu 
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id

    qm_vals, qs_vals = rl.params
    neural_net = get_neural_net_with_prior(qm_vals, 
                                           qs_vals,
                                           initial_prior_var)
    model = BayesianCNN(
        x_train, y_train, neural_net,
        mc_sample=None, learning_rate=lr, beta=beta, rng=rng) # 0.001
    model.init_session()
    model.neural_net.summary()

    # train
    (costs, lik_costs, 
     training_accs, val_accs, 
     trainig_ces, val_ces) = model.train(
            batch_size=64, no_epochs=150, 
            display_epoch=500, 
            x_val=x_test, y_val=y_test, verbose=False)
    # message
    elbo = model.get_elbo()
    rl.pred_prob = np.exp(elbo)
    # infer posterior
    qm_vals, qs_vals, q_names = model.get_weights()
    rl.params = [qm_vals, qs_vals]
    # prediction
    test_acc, test_softmax, test_ce = model.test_accuracy(x_test, y_test)
    rl.test_pred = test_softmax
    rl.test_acc = test_acc
    rl.test_ce = test_ce

    return rl

def run_length_multiproc(
        first_gpu, 
        max_gpu,
        prune_k,
        datagen,
        rng,
        save_path,
        max_iter,
        initial_prior_var,
        beta,
        lr,
        hazard=1./3, # for image classification
        get_neural_net_with_prior=get_bayesian_neural_net_with_prior,
        param_layers_at_most=100,
        surrogate_initial_prior_path=None):
    if os.path.exists(save_path + 'test_accs.npy'):
        test_accs = list(
            np.load(save_path + 'test_accs.npy', allow_pickle=True))
    else:
        test_accs = []

    def _get_cp(prob=1.):
        if surrogate_initial_prior_path is None:
            print("THE INITIAL PRIOR SUPPORT AT MOST "
                  f"{param_layers_at_most}-LAYER NETWORK.")
            qm_vals = [None] * param_layers_at_most
            qs_vals = [None] * param_layers_at_most
        else:
            print("USE SURROGATE PRIOR DISTRIBUTION.")
            qm_vals, qs_vals = load_weights(
            infile_name=surrogate_initial_prior_path)
        return RunLength(0, prob, [qm_vals, qs_vals])

    run_lens = []
    for task_id in range(max_iter):
        # get data
        x_train, y_train, x_test, y_test = datagen.next_task()

        print_log("task: ", task_id)

        # get new run length with length 0
        if not run_lens:
            cp = _get_cp()
        else:
            cp_prob = 0.
            for rl in run_lens:
                rl.prob *= (1-hazard)
                cp_prob += rl.prob*hazard
            cp = _get_cp(prob=cp_prob)
        run_lens.append(cp)

        # split tasks
        results = []
        num_hypotheses = len(run_lens)
        i = 0
        while i < num_hypotheses:
            i += max_gpu
            _hypotheses = run_lens[i-max_gpu:i]
            # args for _process()
            args = []
            def _arg(hypothesis):
                return [hypothesis, x_train, y_train, x_test, y_test,
                    rng, first_gpu, max_gpu,
                    initial_prior_var, lr, beta,
                    get_neural_net_with_prior]
            for hypothesis in _hypotheses:
                args.append(_arg(hypothesis))
            with Pool(np.min((len(_hypotheses), max_gpu))) as p:
                res = p.starmap(_process, args)
                p.close()
            results += res
        run_lens = results


        # calcualte unnormalized growth probability and unnormalized changepoint probability
        normalizer = 0.
        for rl in run_lens:
            rl.r += 1
            rl.prob = rl.prob*rl.pred_prob
            normalizer += rl.prob

        # prune
        run_lens, normalizer = prune(run_lens, normalizer, res_num=prune_k)
        print_log("Selected run length after pruning in order:", [rl.r for rl in run_lens])

        # normalization
        for rl in run_lens:
            rl.prob /= normalizer

        # print log
        print_log("Accuracy and cross entropy:", run_lens[0].test_acc, run_lens[0].test_ce)
        print_log("Task %d ends" % task_id)

        # store most likely result
        test_accs.append(run_lens[0].test_acc)
        np.save(save_path + 'test_accs.npy', test_accs)

    return save_path + 'test_accs.npy'



import argparse
parser = argparse.ArgumentParser(description='Experiment configurations.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='svhn, cifar10')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--first_gpu', metavar='N', type=int, default=0,
                    help='first gpu to be used in nvidia-smi')
parser.add_argument('--num_gpu', metavar='N', type=int, default=1,
                    help=('how many gpus to be used. For example,'
                          '   first_gpu = 2 '
                          '   max_gpu = 6 '
                          ' corresponds to set CUDA_VISIBLE_DEVICES={2,3,4,5,6,7}'
                          '\n'
                          'better to be multiples of beam size.'))
parser.add_argument('--prune_k', metavar='N', type=int, default=3,
                    help='beam size')


if __name__ == '__main__':
    args = parser.parse_args()

    tf.reset_default_graph()
    random_seed = 1
    rng = np.random.RandomState(seed=random_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))
    rng_for_model = np.random.RandomState(seed=random_seed)

    # For example
    #   first_gpu = 0 
    #   max_gpu = 7 
    # corresponds to set CUDA_VISIBLE_DEVICES={0,1,2,3,4,5,6}

    first_gpu = args.first_gpu 
    max_gpu = args.num_gpu 

    # experimental settings
    max_iter = 100
    changerate = 3
    task_size = 20000 
    validation = args.validation

    # algorithm-specific params
    hazard = 0.3
    prune_k = args.prune_k
    initial_prior_var = 1. 
    beta = 1. 
    epoches = 150 
    mini_batch_size = 64

    svhn = False

    if args.dataset == 'svhn':
        # surrogate prior
        surrogate_initial_prior_path = (
            "./meta_prior/svhn_prior_vcl.pkl")

        datagen = LongTransformedSvhnGenerator(
            rng=rng, changerate=changerate, 
            max_iter=max_iter, task_size=task_size,
            validation=validation)

        lr = 0.00025

        folder_name = ("./svhn_with_initial" 
            f'/bocd_res_hazard{hazard}_k{prune_k}'
            f'{"_test" if not validation else ""}'
            '/')

    elif args.dataset == 'cifar10':
        surrogate_initial_prior_path = (
            "./meta_prior/cifar_prior_vcl.pkl")

        datagen = LongTransformedCifar10Generator(
            rng=rng, changerate=changerate, 
            max_iter=max_iter, task_size=task_size,
            validation=validation)

        lr = 0.0005 

        folder_name = ("./cifar10_with_initial" 
            f'/bocd_res_hazard{hazard}_k{prune_k}'
            f'{"_test" if not validation else ""}'
            '/')

    else:
        raise NotImplementedError

    os.makedirs(folder_name)
    # sys.stdout = open(
    #     folder_name + f'log.txt', 
    #     'a') # log file
    print_log('pid = %d' % os.getpid())

    run_length_multiproc(
        first_gpu=first_gpu,
        max_gpu=max_gpu,
        prune_k=prune_k,
        datagen=datagen,
        rng=rng_for_model,
        save_path=folder_name,
        max_iter=max_iter,
        initial_prior_var=initial_prior_var,
        beta=beta,
        lr=lr,
        hazard=hazard, 
        get_neural_net_with_prior=get_bayesian_neural_net_with_prior,
        param_layers_at_most=100,
        surrogate_initial_prior_path=surrogate_initial_prior_path)
