import os
import sys
import numpy as np

from src.game import Game
from src.player_track import PlayerTracking

from src.bocd import BOCD_PlayerTracking
from src.vbs import print_log, BeamSearchHelper, get_beam_search_helper

def vbs_filter(pos_set, config):
    sigma_n2 = config['sigma_n2']
    beam_size = config['vbs']['K']
    beta = config['vbs']['beta'][beam_size]
    p = config['vbs']['p'][beam_size]

    x_olds, x_curs, x_news = pos_set[:-2], pos_set[1:-1], pos_set[2:]
    folder_name = f'./ckpt/ckpt_vbs_p{p}_beta{beta}_sigma{sigma_n2}_b{beam_size}/'
    prior_logodds = np.log(p/(1-p))
    bsh = get_beam_search_helper(folder_name + '/helper.pkl', 
                                 save_folder=folder_name, 
                                 beam_size=beam_size, 
                                 diffusion=beta, 
                                 jump_bias=prior_logodds,
                                 permanent_store_every_tasks=40,
                                 verbose=False)

    logp_preds, logp_preds_1st = [], []
    for task_id, (x_old, x_cur, x_new) in enumerate(zip(x_olds, x_curs, x_news)):
        if (task_id+1) % 2000 == 0:
            print('='*75)
            print_log(f"Start task {task_id}.")

        if task_id <= bsh.task_id:
            print_log(f"Task {task_id} is already trained. Skip training.")
            continue

        hypotheses = bsh.get_new_hypotheses_args(x_old, x_cur, x_new)

        param_set, elbos, test_y_preds = [], [], []
        num_hypotheses = len(hypotheses)
        for hypothesis in hypotheses:
            (model_name,
            s_t,
            diffusion,
            [mu, Lambda],
            x_old, 
            x_cur, 
            x_new) = hypothesis

            # initialization
            track_model = PlayerTracking(sigma_n2=sigma_n2, mu=mu, Lambda=Lambda)

            # broaden if required
            if s_t == 1:
                track_model.broaden_temper(diffusion)

            # compute predictive probability
            log_marginal_likelihood = track_model.log_marginal_likelihood(x_old, x_cur)
            elbos.append(log_marginal_likelihood)

            # absorb new evidence
            track_model.update(x_old, x_cur, compute_cov=True)

            # save weights
            param = track_model.get_params()
            param_set.append(param)

            test_y_preds.append(track_model.log_marginal_likelihood(x_cur, x_new))


        bsh.absorb_new_evidence_and_prune_beams(elbos, param_set)

        test_y_pred = bsh.weighted_test_probability(test_y_preds)
        logp_preds.append(test_y_pred)
        # most likely probability
        test_y_pred_1st = test_y_preds[bsh._indices[0]]
        logp_preds_1st.append(test_y_pred_1st)

    return logp_preds_1st, logp_preds


def bocd_filter(pos_set, config):
    sigma_n2 = config['sigma_n2']
    hazard = config['bocd']['hazard']
    res_num = config['bocd']['K']

    x_olds, x_curs, x_news = pos_set[:-2], pos_set[1:-1], pos_set[2:]
    logp_preds, logp_preds_1st = [], []

    bocd_helper = BOCD_PlayerTracking(
        hazard=hazard, 
        res_num=res_num)

    logp_preds, logp_preds_1st = [], []
    for i, (x_old, x_cur, x_new) in enumerate(zip(x_olds, x_curs, x_news)):
        # add a new run length hypothesis
        bocd_helper.add_new_cp_hypo()

        # evaluate each run length
        for rl in bocd_helper.run_lens:
            mu, Lambda = rl.params
            track_model = PlayerTracking(sigma_n2=sigma_n2, mu=mu, Lambda=Lambda)
            logp_pred = track_model.log_marginal_likelihood(x_old, x_cur)
            rl.pred_prob = np.exp(logp_pred)
            # infer posterior distributions and update
            track_model.update(x_old, x_cur, compute_cov=True)
            rl.params = track_model.get_params()
            # prediction
            rl.test_pred = track_model.log_marginal_likelihood(x_cur, x_new)

        # rank and prune
        bocd_helper.step()

        # evaluation
        # ensemble prediction: a lower bound
        test_y_pred = np.sum([rl.prob*rl.test_pred 
                              for rl in bocd_helper.run_lens])
        logp_preds.append(test_y_pred)
        # most likely probability
        test_y_pred_1st = bocd_helper.run_lens[0].test_pred
        logp_preds_1st.append(test_y_pred_1st)

    return logp_preds_1st, logp_preds


def bf_filter(pos_set, config):
    sigma_n2 = config['sigma_n2']
    beta = config['bf']['beta']

    x_olds, x_curs, x_news = pos_set[:-2], pos_set[1:-1], pos_set[2:]
    track_model = PlayerTracking(sigma_n2=sigma_n2)
    logp_preds = []
    for i, (x_old, x_cur, x_new) in enumerate(zip(x_olds, x_curs, x_news)):
        # update
        track_model.update(x_old, x_cur, compute_cov=True)
        # broaden
        track_model.broaden_Bayesian_forget(beta)
        # predictvie probability
        logp_pred = track_model.log_marginal_likelihood(x_cur, x_new)
        # record
        logp_preds.append(logp_pred)

    return logp_preds


def vcl_filter(pos_set, config):
    sigma_n2 = config['sigma_n2']

    x_olds, x_curs, x_news = pos_set[:-2], pos_set[1:-1], pos_set[2:]
    track_model = PlayerTracking(sigma_n2=sigma_n2)
    logp_preds = []
    for i, (x_old, x_cur, x_new) in enumerate(zip(x_olds, x_curs, x_news)):
        # update
        track_model.update(x_old, x_cur, compute_cov=True)
        # predictvie probability
        logp_pred = track_model.log_marginal_likelihood(x_cur, x_new)
        # record
        logp_preds.append(logp_pred)

    return logp_preds


def ib_filter(pos_set, config):
    sigma_n2 = config['sigma_n2']

    x_olds, x_curs, x_news = pos_set[:-2], pos_set[1:-1], pos_set[2:]
    logp_preds = []
    for i, (x_old, x_cur, x_new) in enumerate(zip(x_olds, x_curs, x_news)):
        track_model = PlayerTracking(sigma_n2=sigma_n2)
        # update
        track_model.update(x_old, x_cur, compute_cov=True)
        # predictvie probability
        logp_pred = track_model.log_marginal_likelihood(x_cur, x_new)
        # record
        logp_preds.append(logp_pred)

    return logp_preds