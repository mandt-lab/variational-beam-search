import abc
import numpy as np
from scipy import stats


class RunLength:
    def __init__(self, r, prob, params):
        self.r = r
        self.prob = prob
        self.params = params
        self.pred_prob = None # current evidence
        self.factor = None # self.prob*self.pred_prob
        self.test_pred = None # predictions for the next observation


class BOCDHelper(abc.ABC):
    def __init__(self, hazard, res_num=1):
        self.hazard = hazard
        self.res_num = res_num
        self.run_lens = []

    @abc.abstractmethod
    def new_changepoint_hypo(self, cp_prob):
        return

    def prune(self, run_lens, normalizer, res_num=1):
        run_lens.sort(reverse=True, key=lambda x: x.prob)
        if len(run_lens) <= res_num:
            return run_lens, normalizer
            
        for rl in run_lens[res_num:]:
            normalizer -= rl.prob
        run_lens = run_lens[:res_num]
        return run_lens, normalizer

    def add_new_cp_hypo(self):
        # add a changepoint for the next task
        if not self.run_lens:
            cp = self.new_changepoint_hypo(1.)
        else:
            cp_prob = 0.
            for rl in self.run_lens:
                rl.prob *= (1-self.hazard)
                cp_prob += rl.prob*self.hazard
            cp = self.new_changepoint_hypo(cp_prob)
        self.run_lens.append(cp)

    def step(self):
        # evaluate predictive probability for each run length
        for rl in self.run_lens:
            # make sure predictive probabilities and parameters are  registered
            assert rl.pred_prob is not None
            assert rl.params is not None
            
        # calcualte unnormalized growth probability and unnormalized changepoint probability
        normalizer = 0.
        for rl in self.run_lens:
            rl.r += 1
            rl.prob = rl.prob*rl.pred_prob
            normalizer += rl.prob

        # prune
        self.run_lens, normalizer = self.prune(self.run_lens, normalizer, 
                                               res_num=self.res_num)
        # print("most probable run length:", self.run_lens[0].r)

        # normalization
        for rl in self.run_lens:
            rl.prob /= normalizer


class BOCD_Gauss(BOCDHelper):
    def new_changepoint_hypo(self, cp_prob):
        return RunLength(0, cp_prob, [0., 1.])


class BOCD_BayesianLinearRegression(BOCDHelper):
    def __init__(self, num_feature, hazard, res_num=1):
        self.num_feature = num_feature
        super(BOCD_BayesianLinearRegression, self).__init__(hazard=hazard,
                                                            res_num=res_num)

    def new_changepoint_hypo(self, cp_prob):
        return RunLength(0, cp_prob, [None, np.eye(self.num_feature)])


class BOCD_PlayerTracking(BOCDHelper):
    def new_changepoint_hypo(self, cp_prob):
        return RunLength(0, cp_prob, [None, None])


def bocd(hazard, res_num):
    run_lens = [RunLength(0, 1., [0., 1.])] 
    for i, _x in enumerate(x):
        print("task:", i)
        # evaluate predictive probability for each run length
        for rl in run_lens:
            m, s = rl.params
            gauss = Gauss(m=m, s=s)
            rl.pred_prob = gauss.pred_prob(_x)
            # infer posterior distributions and update
            gauss.update(_x)
            rl.params = [gauss.m, gauss.s]
            
        # calcualte unnormalized growth probability and unnormalized changepoint probability
        normalizer = 0.
        for rl in run_lens:
            rl.r += 1
            rl.prob = rl.prob*rl.pred_prob
            normalizer += rl.prob

        # prune
        # greedy search
        run_lens, normalizer = prune(run_lens, normalizer, res_num=100)
        print("most probable run length:", run_lens[0].r)

        # normalization
        for rl in run_lens:
            rl.prob /= normalizer
            # add to posterior probabilitiy array
            post_prob[i, rl.r] = rl.prob

        # add a changepoint for the next task
        cp_prob = 0.
        for rl in run_lens:
            rl.prob *= (1-hazard)
            cp_prob += rl.prob*hazard
        cp = RunLength(0, cp_prob, [0, 1])
        run_lens.append(cp)