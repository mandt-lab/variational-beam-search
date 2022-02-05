import os
import sys
import numpy as np

from bocd import BOCD_BayesianLinearRegression
from vbs import print_log, BeamSearchHelper, get_beam_search_helper
from bayes_linear_regression import BayesLinReg

def vbs_filter(x_train_set, y_train_set, x_test_set, y_test_set, config):
    dataset = config['dataset']
    num_feature = config['num_feature']
    sigma_n = config['sigma_n']
    beam_size = config['vbs']['K']
    beta = config['vbs']['beta'][beam_size]
    p = config['vbs']['p'][beam_size]

    folder_name = f'./{dataset}_ckpt_vbs_p{p}_beta{beta}_sigma{sigma_n}_b{beam_size}/'
    os.mkdir(folder_name)

    prior_logodds = np.log(p/(1-p))
    sigma_p = np.eye(num_feature)
    bsh = get_beam_search_helper(folder_name + '/helper.pkl', 
                                 save_folder=folder_name, 
                                 beam_size=beam_size, 
                                 diffusion=beta, 
                                 jump_bias=prior_logodds)
    
    abs_errs = []
    abs_1st_errs = []
    for task_id, (x_train, y_train, x_test, y_test) in enumerate(
                                                            zip(x_train_set, 
                                                                y_train_set, 
                                                                x_test_set, 
                                                                y_test_set)):
        if task_id % 2000 == 0:
            print('='*75)
            print(f"Start task {task_id}.")

        if task_id <= bsh.task_id:
            print(f"Task {task_id} is already trained. Skip training.")
            continue

        hypotheses = bsh.get_new_hypotheses_args(x_train, y_train, x_test, y_test)

        mu_set, Lambda_set, elbos, test_y_preds = [], [], [], []
        num_hypotheses = len(hypotheses)
        for hypothesis in hypotheses:
            (model_name,
            s_t,
            diffusion,
            mu, 
            Lambda,
            x_train,
            y_train,
            x_test,
            y_test) = hypothesis

            if dataset == 'sensordrift':
                '''This procedure outputs the correct model evidence:
                p(x|s) = int p(x|w) p(w|s) dw
                '''

                # compile the args
                if mu is None:
                    mu = np.zeros(num_feature)
                if Lambda is None:
                    Lambda = np.eye(num_feature)

                 # initialization
                linreg = BayesLinReg(num_feature=num_feature, 
                                     sigma_n=sigma_n, 
                                     mu_0=mu, 
                                     Lambda=Lambda)
                
                # broaden if required
                if s_t == 1:
                    linreg.broaden_Bayesian_forget(diffusion) # beta = diffusion
                
                # compute predictive probability
                marginal_likelihood = linreg.marginal_likelihood(x_train, y_train)
                if marginal_likelihood == 0:
                    # computation stability
                    raise('marginal_likelihood == 0')
                    log_marginal_likelihood = -5
                else:
                    log_marginal_likelihood = np.log(marginal_likelihood)
                elbos.append(log_marginal_likelihood)
                
                # absorb new evidence
                linreg.update(x_train, y_train)
                
                # save weights
                mu_set.append(linreg.mu)
                Lambda_set.append(linreg.Lambda)
                
                # prediction
                test_y_preds.append(linreg.predict_mean(x_test))

            else:
                '''This implementation "reverses" the `update()` and `elbo` 
                (marginal evidence) computation, which counts "twice" the 
                likelihood.

                Note this procedure outputs the model evidence with tempered 
                likelihood:
                p(x|s) propto int p(x|w)^2 p(w|s) dw
                '''
                # compile the args
                if mu is None:
                    mu = np.zeros(num_feature)
                if Lambda is None:
                    Lambda = np.eye(num_feature)

                if s_t == 1:
                    Lambda /= diffusion # beta = 1/diffusion

                # predict and absorb new evidence
                linreg = BayesLinReg(num_feature=num_feature, 
                                     sigma_n=sigma_n, 
                                     mu_0=mu, 
                                     Lambda=Lambda)
                
                linreg.update(x_train, y_train)
                
                # save weights
                mu_set.append(linreg.mu)
                Lambda_set.append(linreg.Lambda)
                
                # compute elbo and predict new samples
                marginal_likelihood = linreg.marginal_likelihood(x_train, y_train)
                if marginal_likelihood == 0:
                    # computation stability
                    log_marginal_likelihood = -5
                else:
                    log_marginal_likelihood = np.log(marginal_likelihood)
                elbos.append(log_marginal_likelihood)
                test_y_preds.append(linreg.predict_mean(x_test))

        bsh.absorb_new_evidence_and_prune_beams(elbos, mu_set, Lambda_set)

        # ensemble prediction
        test_y_pred = bsh.weighted_test_probability(test_y_preds)
        if config['logodds']:
            test_y_pred = 1 / (1 + np.exp(-test_y_pred))
        abs_err = np.abs(test_y_pred - y_test)
        abs_errs.append(abs_err)
        # most likely prediction
        test_y_pred_1st = test_y_preds[bsh._indices[0]]
        if config['logodds']:
            test_y_pred_1st = 1 / (1 + np.exp(-test_y_pred_1st))
        abs_1st_err = np.abs(test_y_pred_1st - y_test)
        abs_1st_errs.append(abs_1st_err)
        
        if task_id % 10000 == 0:
            print('vbs mean error (ensemble):', np.mean(abs_errs))
            print('vbs mean error (most likely):', np.mean(abs_1st_errs))
        
    np.save(folder_name + './abs_errs.npy', abs_errs)
    np.save(folder_name + './abs_1st_errs.npy', abs_1st_errs)
    print('Results saved to', folder_name)

    print('vbs mean error (ensemble):', np.mean(abs_errs))
    print('vbs mean error (most likely):', np.mean(abs_1st_errs))

    return abs_1st_errs, abs_errs


def bocd_filter(x_train_set, y_train_set, x_test_set, y_test_set, config):
    num_feature = config['num_feature']
    sigma_n = config['sigma_n']
    hazard = config['bocd']['hazard']
    res_num = config['bocd']['K']

    dataset = config['dataset']
    folder_name = f'./{dataset}_ckpt_bocd_hazard{hazard}_sigma{sigma_n}_K{res_num}/'
    os.mkdir(folder_name)

    bocd_helper = BOCD_BayesianLinearRegression(
        num_feature=num_feature, 
        hazard=hazard, 
        res_num=res_num)

    abs_errs, abs_1st_errs = [], []
    for i, (x_train, y_train, x_test, y_test) in enumerate(
                                                    zip(x_train_set, 
                                                        y_train_set, 
                                                        x_test_set, 
                                                        y_test_set)):
        # add a new run length hypothesis
        bocd_helper.add_new_cp_hypo()

        # evaluate each run length
        for rl in bocd_helper.run_lens:
            mu, Lambda = rl.params
            linreg = BayesLinReg(num_feature=num_feature, 
                                 sigma_n=sigma_n, 
                                 mu_0=mu, 
                                 Lambda=Lambda)
            pred_prob = linreg.marginal_likelihood(x_train, y_train)
            rl.pred_prob = pred_prob
            # infer posterior distributions and update
            linreg.update(x_train, y_train)
            rl.params = [linreg.mu, linreg.Lambda]
            # prediction
            rl.test_pred = linreg.predict_mean(x_test)

        # rank and prune
        bocd_helper.step()

        # evaluation
        # ensemble prediction
        test_y_pred = np.sum([rl.prob*rl.test_pred 
                                for rl in bocd_helper.run_lens])
        if config['logodds']:
            test_y_pred = 1 / (1 + np.exp(-test_y_pred))
        abs_err = np.abs(test_y_pred - y_test)
        abs_errs.append(abs_err)
        # most likely prediction
        test_y_pred_1st = bocd_helper.run_lens[0].test_pred
        if config['logodds']:
            test_y_pred_1st = 1 / (1 + np.exp(-test_y_pred_1st))
        abs_1st_err = np.abs(test_y_pred_1st - y_test)
        abs_1st_errs.append(abs_1st_err)

    np.save(folder_name + './abs_errs.npy', abs_errs)
    np.save(folder_name + './abs_1st_errs.npy', abs_1st_errs)
    print('Results saved to', folder_name)

    print('bocd mean error (ensemble):', np.mean(abs_errs))
    print('bocd mean error (most likely):', np.mean(abs_1st_errs))
    return abs_1st_errs, abs_errs


def bf_filter(x_train_set, y_train_set, x_test_set, y_test_set, config):
    beta = config['bf']['beta']
    sigma_n = config['sigma_n']

    dataset = config['dataset']
    folder_name = f'./{dataset}_ckpt_bf_beta{beta}_sigma{sigma_n}/'
    os.mkdir(folder_name)

    bayes_linreg = BayesLinReg(num_feature=config['num_feature'], 
                               sigma_p=np.eye(config['num_feature']), 
                               sigma_n=sigma_n)
    abs_errs = []
    for i, (x_train, y_train, x_test, y_test) in enumerate(
                                                        zip(x_train_set, 
                                                            y_train_set, 
                                                            x_test_set, 
                                                            y_test_set)):

        # update
        bayes_linreg.update(x_train, y_train, compute_cov=False)
        # predict
        if config['logodds']:
            y_test_pred = bayes_linreg.logit_predict_map(x_test)
        else:
            y_test_pred = bayes_linreg.predict_mean(x_test)
        # error
        abs_errs.append(np.abs(y_test_pred - y_test))
        # broaden
        bayes_linreg.broaden_Bayesian_forget(beta)

    np.save(folder_name + './abs_errs.npy', abs_errs)
    print('Results saved to', folder_name)

    mean_err = np.mean(abs_errs)
    print('bf mean error:', mean_err)

    return abs_errs


def vcl_filter(x_train_set, y_train_set, x_test_set, y_test_set, config):
    dataset = config['dataset']
    sigma_n = config['sigma_n']
    folder_name = f'./{dataset}_ckpt_vcl_sigma{sigma_n}/'
    os.mkdir(folder_name)

    bayes_linreg = BayesLinReg(num_feature=config['num_feature'], 
                               sigma_p=np.eye(config['num_feature']), 
                               sigma_n=sigma_n)
    abs_errs = []
    for i, (x_train, y_train, x_test, y_test) in enumerate(
                                                        zip(x_train_set, 
                                                            y_train_set, 
                                                            x_test_set, 
                                                            y_test_set)):

        # update
        bayes_linreg.update(x_train, y_train, compute_cov=False)
        # predict
        if config['logodds']:
            y_test_pred = bayes_linreg.logit_predict_map(x_test)
        else:
            y_test_pred = bayes_linreg.predict_mean(x_test)
        # error
        abs_errs.append(np.abs(y_test_pred - y_test))

    np.save(folder_name + './abs_errs.npy', abs_errs)
    print('Results saved to', folder_name)

    mean_err = np.mean(abs_errs)
    print('vcl mean error:', mean_err)

    return abs_errs


def ib_filter(x_train_set, y_train_set, x_test_set, y_test_set, config):
    dataset = config['dataset']
    sigma_n = config['sigma_n']
    folder_name = f'./{dataset}_ckpt_ib_sigma{sigma_n}/'
    os.mkdir(folder_name)

    abs_errs = []
    for i, (x_train, y_train, x_test, y_test) in enumerate(
                                                        zip(x_train_set, 
                                                            y_train_set, 
                                                            x_test_set, 
                                                            y_test_set)):

        bayes_linreg = BayesLinReg(num_feature=config['num_feature'], 
                                   sigma_p=np.eye(config['num_feature']), 
                                   sigma_n=sigma_n)

        # update
        bayes_linreg.update(x_train, y_train, compute_cov=False)
        # predict
        if config['logodds']:
            y_test_pred = bayes_linreg.logit_predict_map(x_test)
        else:
            y_test_pred = bayes_linreg.predict_mean(x_test)
        # error
        abs_errs.append(np.abs(y_test_pred - y_test))

    np.save(folder_name + './abs_errs.npy', abs_errs)
    print('Results saved to', folder_name)

    mean_err = np.mean(abs_errs)
    print('ib mean error:', mean_err)

    return abs_errs