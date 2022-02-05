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
# beam diversification
from util import find_optimal_beam

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BeamSearchHelper:
    """A helper to ensure constant disk usage.

    It uses an internal protocol of hypothses arrangements:
    [
        the first child (0) of the first parent,
        the second child (1)of the first parent,
        the first child (0)of the second parent,
        the second child (1)of the second parent,
        the first child (0)of the third parent,
        the second child (1)of the third parent,
        ...
    ]

    Note that it is worker's reponsibility to load prior weights and load 
    trained weights.
    """
    def __init__(self, save_folder, beam_size, diffusion, jump_bias,
                 max_steps_to_store=10, permanent_store_every_tasks=10,
                 max_decision_length=2000, temper_every_iter=False):
        self.save_folder = save_folder + '/'
        self.beam_size = beam_size
        self.diffusion = diffusion
        self.jump_bias = jump_bias
        self.max_steps_to_store = max_steps_to_store
        self.permanent_store_every_tasks = permanent_store_every_tasks
        self.max_decision_length = max_decision_length
        self.temper_every_iter = temper_every_iter

        print('='*75)
        print("Variational Beam Search:")
        print(f"At most {self.beam_size} are allowed to keep.")
        print(f"Use relative broadening with factor {self.diffusion} if s=1.")
        print(f"Jump bias for change variables is set to {self.jump_bias}")
        print(f"\nFiles and models are saved to {self.save_folder}")
        print(f"It keeps {self.max_steps_to_store} most recent tasks' results.")
        print("It permanently stores the intermediate results every "
              f"{self.permanent_store_every_tasks} tasks.")
        print(f"{self.max_decision_length} most recent decisions are stored. "
              "older decisions are deemed outdated and droped.")
        print('='*75)

        self.task_id = -1
        # previous task results
        # keys = ["log_post_prob", "save_path", "decisions"]
        self.latest_task_res = [] 
        # in-processing current task results
        self.current_task_res = []
        self._cur_hypos_args = None
        self._indices = None
        self._indices_count = None

    def get_new_hypotheses_args(self, *data_wrapper):
        """Will give out necessary hyper arguments for each hypothesis. 
        These include:
        h_i = [
            prior_weights_path,
            save_path,
            s_t,
            diffusion,
            x_train,
            y_train,
            x_test,
            y_test,
        ]

        Arguments:
        data_wrapper -- [x_train, y_train, x_test, y_test]

        Returns:
        A list of lists: 
            hypos_args = [
                args of the first child of the first parent,
                args of the second child of the first parent,
                args of the first child of the second parent,
                args of the second child of the second parent,
                args of the first child of the third parent,
                args of the second child of the third parent,
                ...
            ]
        Every consecutive pair inherits the same parent. For example, h_1 and 
        h_2 keep every parameter the same except s_t taking 0 and 1.
        """
        self.task_id += 1
        print('='*75)
        print_log(f"Get new hypotheses for task {self.task_id}.")
        hypos_args = []
        if not self.latest_task_res:
            # initial task:
            # we set s_0 = 0 because we don't want to broaden the prior dist.
            # but in practice it is regarded as a novel task
            s_t = 0
            hypo_args = [None, 
                         self.save_folder+f"task{self.task_id}_h0_s{s_t}.pkl", 
                         s_t, 
                         self.diffusion, 
                         *data_wrapper]
            hypos_args.append(hypo_args)
        else:
            for h, parent_hypo in enumerate(self.latest_task_res):
                if self.diffusion == 1.:
                    # Variational Continual Learning
                    s_set = [0]
                elif self.temper_every_iter:
                    # tempering for every iteration
                    s_set = [1]
                else: 
                    # normal
                    s_set = [0, 1]
                for s_t in s_set:
                    hypo_args = [
                        parent_hypo["save_path"], 
                        self.save_folder+f"task{self.task_id}_h{h}_s{s_t}.pkl", 
                        s_t, 
                        self.diffusion, 
                        *data_wrapper]
                    hypos_args.append(hypo_args)
        self._cur_hypos_args = hypos_args
        return hypos_args

    def calculate_posterior_probabilities(self, elbos):
        """`elbos` come in the protocol order.
        """
        total_log_prob = None
        if not self.latest_task_res:
            total_log_prob = 0.
        else:
            total_log_prob = np.asarray([parent_hypo["log_post_prob"] 
                                for parent_hypo in self.latest_task_res])
        if self.task_id == 0 or len(elbos) == 1:
            print_log("Single hypothesis: task_id == 0 or len(elbos) == 1")
            log_post_prob = [0.]
        else:
            elbos_0 = np.asarray(elbos[0::2]) # at odd position
            elbos_1 = np.asarray(elbos[1::2]) # at even position
            z = elbos_1 - elbos_0 + self.jump_bias
            # log q(s_t=1) = log (sigmoid(z)) = log (1 / (1 + exp(-z)) =
            # -log(1+exp(-z))
            child_s1_single_log_prob = -np.log1p(np.exp(-z))
            # log q(s_t=0) = log (1 - q(s_t=1)) = log(1 - sigmoid(z)) =
            # log(sigmoid(-z)) = -log(1+exp(z))
            child_s0_single_log_prob = -np.log1p(np.exp(+z))
            # modify -inf to enhance computational stability
            inf_ind = np.argwhere(child_s1_single_log_prob == -np.inf)
            child_s1_single_log_prob[inf_ind[:,0]] = z[inf_ind[:,0]]
            inf_ind = np.argwhere(child_s0_single_log_prob == -np.inf)
            child_s0_single_log_prob[inf_ind[:,0]] = -z[inf_ind[:,0]]
            # if child_s1_single_log_prob == -np.inf:
            #     child_s1_single_log_prob = z
            # if child_s0_single_log_prob == -np.inf:
            #     child_s0_single_log_prob = -z

            # log q(s_{1:t}) = log q(s_t) + log q(s_{i:(t-1)})
            child_s1_total_log_prob = child_s1_single_log_prob + total_log_prob
            child_s0_total_log_prob = child_s0_single_log_prob + total_log_prob

            # assemble
            log_post_prob = [None] * len(elbos)
            log_post_prob[0::2] = child_s0_total_log_prob
            log_post_prob[1::2] = child_s1_total_log_prob
        return log_post_prob

    def prune_beams(self, log_post_prob):
        print_log("Recording and pruning beams...", end='')
        log_post_prob = np.array(log_post_prob)
        indices = find_optimal_beam(log_post_prob, 
                                    self.beam_size, 
                                    discard_fraction=1.0/3.0)

        self.current_task_res = []
        for i, (log_prob, hypo) in enumerate(
                zip(log_post_prob, self._cur_hypos_args)):
            if not self.latest_task_res:
                parent_hypo = {"decisions": ""}
            else:
                parent_hypo = self.latest_task_res[i//2]
            self.current_task_res.append(
                {"save_path": hypo[1],
                 "log_post_prob": log_prob,
                 "decisions": parent_hypo["decisions"] + str(hypo[2])})

        if len(self.current_task_res) > self.beam_size:
            self.current_task_res = [self.current_task_res[i] for i in indices]
        assert len(self.current_task_res) <= self.beam_size
        print("Done")
        self._indices = indices
        self._indices_count = self.task_id
        print_log(f"`indices` are updated to task {self.task_id}.")

        print('-'*75)
        print_log("Sorted models of beams are: ")
        for hypo in self.current_task_res:
            print_log("Model name:", os.path.basename(hypo["save_path"])) 
            print_log("\tLog probability:", hypo["log_post_prob"])
            print_log("\tDecisions so far:", hypo["decisions"])
        print_log(
            "Beams remaining after truncation: "
            f"{len(self.current_task_res)}")
        print('-'*75)

    def weighted_test_probability(self, test_softmaxs):
        if self._indices_count != self.task_id:
            print_log(f"TRYING TO EVALUATE PERFORMANCE ON TASK {self.task_id} "
                      f"WITH OUTDATED MODEL {self._indices_count}.")
        else:
            print_log(f"Evaluate performance on task {self.task_id}.")
        test_softmaxs = [test_softmaxs[i] for i in self._indices]
        unnorm_weights = [np.exp(hypo["log_post_prob"]) 
                            for hypo in self.current_task_res]
        weights = [w / sum(unnorm_weights) for w in unnorm_weights]
        weighted_softmax = sum(
            [softmax * w for (softmax, w) in zip(test_softmaxs, weights)])
        return weighted_softmax

    def update_storage(self):
        """This should not be executed until all relevant parts of current task 
        finish. There is a concept transfer from "current task" to "latest
        task", which only occurs after current task becomes outdated.
        """
        self.latest_task_res = self.current_task_res

        # truncate decision length if necessary
        if (self.task_id+1) > self.max_decision_length:
            for i in range(len(self.latest_task_res)):
                decisions = self.latest_task_res[i]["decisions"]
                self.latest_task_res[i]["decisions"] = \
                    decisions[-self.max_decision_length:]
            print_log(
                "Decision length is truncated for model", 
                os.path.basename(self.latest_task_res[i]["save_path"]))

        # update self
        pickle.dump(
            self, open(self.save_folder+"helper.pkl", "wb"))

        # remove outdated files
        if self.task_id >= self.max_steps_to_store:
            del_task_id = self.task_id - self.max_steps_to_store
            if del_task_id % self.permanent_store_every_tasks == 0:
                # permanently store every M tasks
                pickle.dump(
                    self, 
                    open(self.save_folder+f"helper{self.task_id}.pkl", "wb"))
                print_log(f"Permanently store task {self.task_id} results.")
                return
            num_parents = (
                int(2**(del_task_id-1))
                if 2*self.beam_size > 2**del_task_id 
                else int(self.beam_size))
            if num_parents < 1:
                num_parents = 1
            for h in range(num_parents):
                if self.diffusion == 1.:
                    # Variational Continual Learning
                    s_set = [0]
                elif self.temper_every_iter:
                    # tempering for every iteration
                    s_set = [1]
                else: 
                    # normal
                    s_set = [0, 1]
                for s_t in s_set:
                    file_path = (self.save_folder+
                        f"task{del_task_id}_h{h}_s{s_t}.pkl")
                    print_log(f"Removing file: {file_path}...", end='')
                    os.remove(file_path)
                    print("Done")

# Factory function to return a beam search object
def get_beam_search_helper(helper_path, **kwargs):
    if os.path.isfile(helper_path):
        return pickle.load(open(helper_path, "rb"))
    else:
        return BeamSearchHelper(**kwargs)

def _process(
        # model specific
        prior_weights_path, save_path, s_t, diffusion,
        x_train, y_train, x_test, y_test, 
        # environment
        rng, first_gpu, max_gpu, task_id,
        # optimizaton
        surrogate_initial_prior_path=None,
        initial_prior_var=1.,
        beta=1.,
        lr=0.0005,
        epoches=1000,
        mini_batch_size=64,
        independent=False,
        no_train_samples=10, no_test_samples=20, temper=1,
        display_epoch=100):
    '''worker process
    '''
    # delete unused params
    del no_train_samples, no_test_samples, temper
    del display_epoch

    # set visible gpus
    proc = multiprocessing.current_process()
    proc_id = int(proc.name.split('-')[-1])
    # each process assumes one specific gpu
    gpu_id = first_gpu + proc_id % max_gpu 
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id

    # get priors
    if task_id == 0 or independent:
        if surrogate_initial_prior_path is None:
            param_layers_at_most = 6
            print("THE INITIAL PRIOR SUPPORT AT MOST "
                  f"{param_layers_at_most}-LAYER NETWORK.")
            prior_m = [None] * param_layers_at_most
            prior_s = [None] * param_layers_at_most
        else:
            print("USE SURROGATE PRIOR DISTRIBUTION.")
            prior_m, prior_s = load_weights(
            infile_name=surrogate_initial_prior_path)
    else:
        prior_m, prior_s = load_weights(
            infile_name=prior_weights_path)
        if s_t == 1:
            prior_m, prior_s = broaden_weights(
                prior_m, prior_s, diffusion)

    neural_net = get_bayesian_neural_net_with_prior(
        prior_m, 
        prior_s, 
        initial_prior_var=initial_prior_var)

    result = {}
    model = BayesianCNN(
        x_train, y_train, neural_net,
        mc_sample=None, learning_rate=lr, beta=beta, rng=rng)
    model.init_session()
    # model.neural_net.summary()

    # train
    (costs, lik_costs, 
     training_accs, val_accs, 
     trainig_ces, val_ces) = model.train(
            batch_size=mini_batch_size, no_epochs=epoches, display_epoch=500, 
            x_val=x_test, y_val=y_test, verbose=False)
    np.save(save_path[:-4] + "_train_info.npy", 
        [costs, lik_costs, 
         training_accs, val_accs, 
         trainig_ces, val_ces])

    qm_vals, qs_vals, q_names = model.get_weights()
    save_weights([qm_vals, qs_vals], 
        outfile_name=save_path)

    print_log("Getting ELBO:")
    elbo = model.get_elbo()
    print_log(f"\t{elbo}")

    # test
    test_acc, test_softmax, test_ce = model.test_accuracy(
        x_test, y_test)
    
    acc = np.zeros(max_iter) # vector for storage
    acc[task_id] = test_acc
    print_log(
        f"Hypothesis {os.path.basename(save_path)} testing done. "
        f"Accuracy: {test_acc}")

    result["elbo"] = elbo
    result["test_metrics"] = test_acc #acc
    result["test_softmax"] = test_softmax
    return result

def multiproc_vbs(
        datagen,
        first_gpu,
        num_gpu,
        rng,
        save_path,
        beam_size,
        diffusion,
        jump_bias,
        max_iter,
        changerate,
        mult_diff,
        restart_every_iter,
        initial_prior_var,
        beta,
        lr,
        epoches,
        mini_batch_size,
        surrogate_initial_prior_path,
        temper_every_iter):
    """We mainly use this funciton for the fast inference after the optimal 
    hyperparameter setting is known. For hyperparamter tuning, use 
    singleproc_vbs().

    Args:
    diffusion -- It will be taken the square root to act on the standard 
        deviation.
    """
    del mult_diff # unused arguments

    diffusion = np.sqrt(diffusion)

    if os.path.exists(save_path + '/test_accs.npy'):
        test_accs = np.load(save_path + '/test_accs.npy')
    else:
        test_accs = np.zeros(max_iter) # save test results

    helper_path = (save_path + '/'
        + "helper.pkl")

    bsh = get_beam_search_helper(
        helper_path=helper_path,
        save_folder=save_path, 
        beam_size=beam_size, 
        diffusion=diffusion, 
        jump_bias=jump_bias,
        max_steps_to_store=10, 
        permanent_store_every_tasks=10,
        max_decision_length=2000,
        temper_every_iter=temper_every_iter
    )
    print_log("Beam search helper initialized.")

    for task_id in range(max_iter):
        print('='*75)
        print_log(f"Start task {task_id}.")
        if task_id % changerate == 0:
            print_log(f"Switch point on this task ({task_id})!")
        x_train, y_train, x_test, y_test = datagen.next_task()
        print_log(f"Finish reading data for task {task_id}")

        if task_id <= bsh.task_id:
            print_log(f"Task {task_id} is already trained. Skip training.")
            continue

        hypotheses = bsh.get_new_hypotheses_args(
            x_train, y_train, x_test, y_test)

        results = []
        num_hypotheses = len(hypotheses)
        i = 0
        while i < num_hypotheses:
            i += num_gpu
            _hypotheses = hypotheses[i-num_gpu:i]
            # args for _process()
            args = []
            def _arg(hypothesis):
                return hypothesis + [
                    rng, first_gpu, num_gpu, task_id,
                    surrogate_initial_prior_path,
                    initial_prior_var,
                    beta,
                    lr,
                    epoches,
                    mini_batch_size,
                    restart_every_iter,
                ]
            for hypothesis in _hypotheses:
                args.append(_arg(hypothesis))
            with Pool(np.min((len(_hypotheses), num_gpu))) as p:
                res = p.starmap(_process, args)
                p.close()
            results += res

        elbos, accs, test_softmaxs = [], [], []
        for res in results:
            elbos.append(res["elbo"])
            accs.append(res["test_metrics"])
            test_softmaxs.append(res["test_softmax"])
        log_post_prob = bsh.calculate_posterior_probabilities(elbos)
        bsh.prune_beams(log_post_prob)
        test_softmax = bsh.weighted_test_probability(test_softmaxs)
        indices = bsh._indices
        np.save(save_path+f"/acc_task{task_id}.npy", accs)
        np.save(save_path+f"/elbo_task{task_id}.npy", elbos)
        np.save(save_path+f"/softmax_task{task_id}.npy", test_softmaxs)
        np.save(save_path+f"/indices_task{task_id}.npy", indices)
        # final accuracy
        test_pred = np.argmax(test_softmax, axis=1)
        test_accuracy = np.sum((test_pred - y_test) == 0)/y_test.shape[0]
        test_accs[task_id] = test_accuracy
        np.save(save_path+"/test_accs.npy", test_accs)
        print_log(f"Task {task_id} test accuracy: {test_accuracy}")

        bsh.update_storage()

    return test_accs


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
parser.add_argument('--beam_size', metavar='N', type=int, default=1,
                    help='beam size')


if __name__ == '__main__':
    args = parser.parse_args()

    tf.reset_default_graph()
    random_seed = 1
    rng = np.random.RandomState(seed=random_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))
    rng_for_model = np.random.RandomState(seed=random_seed)

    # For example
    #   first_gpu = 2 
    #   max_gpu = 6 
    # corresponds to set CUDA_VISIBLE_DEVICES={2,3,4,5,6,7}

    first_gpu = args.first_gpu 
    max_gpu = args.num_gpu 

    # experimental settings
    max_iter = 100
    changerate = 3
    task_size = 20000 
    validation = args.validation

    # algorithm-specific params
    beam_size = args.beam_size
    temper_every_iter = False 
    restart_every_iter = False 
    mult_diff = True
    diffusion = 1.5 
    jump_bias = 0.

    initial_prior_var = 1. 
    beta = 1. 
    # lr = 0.00025 
    epoches = 150 
    mini_batch_size = 64

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
            f'/vbs_res_d{diffusion}_beam{beam_size}'
            f'{"_test" if not validation else ""}'
            f'{"_restart_every_iter" if restart_every_iter else ""}'
            f'{"_temper_every_iter" if temper_every_iter else ""}'
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
            f'/vbs_res_d{diffusion}_beam{beam_size}'
            f'{"_test" if not validation else ""}'
            f'{"_restart_every_iter" if restart_every_iter else ""}'
            f'{"_temper_every_iter" if temper_every_iter else ""}'
            '/')

    else:
        raise NotImplementedError

    os.makedirs(folder_name)
    # sys.stdout = open(
    #     folder_name + f'log_b{beam_size}.txt', 
    #     'a') # log file
    print_log('pid = %d' % os.getpid())
    
    multiproc_vbs(
        datagen=datagen,
        rng=rng_for_model,
        first_gpu=first_gpu,
        num_gpu=max_gpu,
        save_path=folder_name,
        beam_size=beam_size,
        diffusion=diffusion,
        jump_bias=jump_bias,
        max_iter=max_iter,
        changerate=changerate,
        mult_diff=mult_diff,
        restart_every_iter=restart_every_iter,
        initial_prior_var=initial_prior_var,
        beta=beta,
        lr=lr,
        epoches=epoches,
        mini_batch_size=mini_batch_size,
        surrogate_initial_prior_path=surrogate_initial_prior_path,
        temper_every_iter=temper_every_iter)
