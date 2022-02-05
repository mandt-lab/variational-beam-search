import sys
import os
from datetime import datetime
import pickle
import itertools
import numpy as np

def print_log(*args, **kwargs):
    print("[{}]".format(datetime.now()), *args, **kwargs)
    sys.stdout.flush()

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
                 max_steps_to_store=10, permanent_store_every_tasks=2000,
                 max_decision_length=2000, temper_every_iter=False):
        self.save_folder = save_folder + '/'
        self.beam_size = beam_size
        self.diffusion = diffusion
        self.jump_bias = jump_bias
        self.max_steps_to_store = max_steps_to_store
        self.permanent_store_every_tasks = permanent_store_every_tasks
        self.max_decision_length = max_decision_length
        self.temper_every_iter = temper_every_iter

        os.makedirs(self.save_folder, exist_ok=True)

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

    def print_log(self, *args, **kwargs):
        if self.task_id % self.permanent_store_every_tasks == 0:
            print("[{}]".format(datetime.now()), *args, **kwargs)
            sys.stdout.flush()

    def print(self, *args, **kwargs):
        if self.task_id % self.permanent_store_every_tasks == 0:
            print(*args, **kwargs)
            sys.stdout.flush()

    def get_new_hypotheses_args(self, *data_wrapper):
        """Will give out necessary hyper arguments for each hypothesis. 
        These include:
        h_i = [
            model_name,
            s_t,
            diffusion,
            mu, 
            Lambda,
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
        self.print('='*75)
        self.print_log(f"Get new hypotheses for task {self.task_id}.")
        hypos_args = []
        if not self.latest_task_res:
            # initial task:
            # we set s_0 = 0 because we don't want to broaden the prior dist.
            # but in practice it is regarded as a novel task
            s_t = 0
            hypo_args = ['0',
                         s_t, 
                         self.diffusion, 
                         None, # mu
                         None, # Lambda
                         *data_wrapper]
            hypos_args.append(hypo_args)
        else:
            for h, parent_hypo_res in enumerate(self.latest_task_res):
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
                        str(h),
                        s_t, 
                        self.diffusion, 
                        parent_hypo_res['mu'],
                        parent_hypo_res['Lambda'],
                        *data_wrapper]
                    hypos_args.append(hypo_args)
        self._cur_hypos_args = hypos_args
        return hypos_args

    def absorb_new_evidence_and_prune_beams(self, elbos, mu_set, Lambda_set):
        log_post_prob = self._calculate_posterior_probabilities(elbos)
        self._prune_beams(mu_set, Lambda_set, log_post_prob)
        self._update_storage()

    def _calculate_posterior_probabilities(self, elbos):
        """`elbos` come in the protocol order.
        """
        total_log_prob = None
        if not self.latest_task_res:
            total_log_prob = 0.
        else:
            total_log_prob = np.asarray([parent_hypo_res["log_post_prob"] 
                                for parent_hypo_res in self.latest_task_res])
        if self.task_id == 0 or len(elbos) == 1:
            self.print_log("Single hypothesis: task_id == 0 or len(elbos) == 1")
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

            # log q(s_{1:t}) = log q(s_t) + log q(s_{i:(t-1)})
            child_s1_total_log_prob = child_s1_single_log_prob + total_log_prob
            child_s0_total_log_prob = child_s0_single_log_prob + total_log_prob

            # assemble
            log_post_prob = [None] * len(elbos)
            log_post_prob[0::2] = child_s0_total_log_prob
            log_post_prob[1::2] = child_s1_total_log_prob
        return log_post_prob

    def _prune_beams(self, mu_set, Lambda_set, log_post_prob):
        self.print_log("Recording and pruning beams...", end='')
        log_post_prob = np.array(log_post_prob)
        # diversified beam
        indices = find_optimal_beam(log_post_prob, 
                                    self.beam_size, 
                                    discard_fraction=1.0/3.0)
        # original beam
        # indices = np.argsort(-log_post_prob)[:self.beam_size]

        # update current task
        self.current_task_res = []
        for i, (mu, Lambda, log_prob, hypo) in enumerate(
                zip(mu_set, Lambda_set, log_post_prob, self._cur_hypos_args)):
            if not self.latest_task_res:
                parent_hypo_res = {"decisions": ""}
            else:
                parent_hypo_res = self.latest_task_res[i//2]
            self.current_task_res.append(
                {"model_name": hypo[0] + '_' + str(hypo[1]),
                 "mu": mu,
                 "Lambda": Lambda,
                 "log_post_prob": log_prob,
                 "decisions": parent_hypo_res["decisions"] + str(hypo[1])})
            self.print_log(
                "Model name:", hypo[0] + '_' + str(hypo[1]))
            self.print_log("\tLog probability:", log_prob)

        if len(self.current_task_res) > self.beam_size:
            self.current_task_res = [self.current_task_res[i] for i in indices]
        assert len(self.current_task_res) <= self.beam_size
        self.print("Done")

        self._indices = indices
        self._indices_count = self.task_id
        self.print_log(f"`indices` are updated to task {self.task_id}.")

        self.print('-'*75)
        self.print_log("Sorted models of beams are: ")
        for hypo_res in self.current_task_res:
            self.print_log(
                "Model name:", hypo_res["model_name"])
            self.print_log("\tLog probability:", hypo_res["log_post_prob"])
            self.print_log("\tDecisions so far:", hypo_res["decisions"])
        self.print_log(
            "Beams remaining after truncation: "
            f"{len(self.current_task_res)}")
        self.print('-'*75)

    def weighted_test_probability(self, predictive_probs):
        if self._indices_count != self.task_id:
            print_log(f"TRYING TO EVALUATE PERFORMANCE ON TASK {self.task_id} "
                      f"WITH OUTDATED MODEL {self._indices_count}.")
        else:
            self.print_log(f"Evaluate performance on task {self.task_id}.")
        predictive_probs = [predictive_probs[i] for i in self._indices]
        unnorm_weights = [np.exp(hypo_res["log_post_prob"]) 
                            for hypo_res in self.current_task_res]
        if sum(unnorm_weights) == 0.:
            unnorm_weights = np.ones_like(unnorm_weights)
        weights = [w / sum(unnorm_weights) for w in unnorm_weights]
        weighted_softmax = sum(
            [softmax * w for (softmax, w) in zip(predictive_probs, weights)])
        return weighted_softmax

    def _update_storage(self):
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
            self.print_log(
                "Decision length is truncated for model", 
                self.latest_task_res[i]["model_name"])

        # update self
        pickle.dump(
            self, open(self.save_folder+"helper.pkl", "wb"))

        # permanently store every M tasks:
        if self.task_id % self.permanent_store_every_tasks == 0:
            # permanently store every M tasks
            pickle.dump(
                self, 
                open(self.save_folder+f"helper{self.task_id}.pkl", "wb"))
            self.print_log(f"Permanently store task {self.task_id} results.")
                    
    def save_weights(self, weights_list, outfile_name='weights.pkl'):
        # save with the binary protocol
        with open(outfile_name, 'wb') as outfile:
            pickle.dump(weights_list, outfile, pickle.HIGHEST_PROTOCOL)
            self.print_log('save weights in ' + outfile_name)

    def load_weights(self, infile_name='weights.pkl'):
        # save with the binary protocol
        self.print_log('load weights from ' + infile_name)
        with open(infile_name, 'rb') as infile:
            weights_list = pickle.load(infile)
        return weights_list

# Factory function to return a beam search object
def get_beam_search_helper(helper_path, **kwargs):
    if os.path.isfile(helper_path):
        # TODO: haven't tested this funciton yet
        return pickle.load(open(helper_path, "rb"))
    else:
        return BeamSearchHelper(**kwargs)