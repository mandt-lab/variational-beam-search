import os
import sys
import numpy as np
import argparse

from filters import vbs_filter, bocd_filter, bf_filter, vcl_filter, ib_filter


config = {
    'sigma_n2': 0.1,
    'bocd': {
        'K': 6, # ensemble size 
        'hazard': 0.99,
    },
    'bf': {
        'beta': 0.9,
    },
    'vbs': {
        'K': 1, # ensemble size
        'beta': {
            1: 0.5,
            3: 0.6,
            6: 0.7,
        }, 
        'p': {
            1: 0.513,
            3: 0.507,
            6: 0.505,
        },
    },
}

parser = argparse.ArgumentParser(description='Experiment configurations.')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--filter', type=str, default='vbs')


if __name__ == "__main__":

    args = parser.parse_args()

    meta_tr_pos_set = np.load('data/meta_tr_pos_set_conn.npy', allow_pickle=True)
    meta_te_pos_set = np.load('data/meta_te_pos_set_conn.npy', allow_pickle=True)


    meta_pos_set = None
    if args.validation:
        meta_pos_set = meta_tr_pos_set
    else:
        meta_pos_set = meta_te_pos_set


    meta_logp_preds = []
    for pos_set in meta_pos_set:
        if args.filter == 'vbs':
            logp_preds, _ = vbs_filter(pos_set, config)
        elif args.filter == 'bf':
            logp_preds = bf_filter(pos_set, config)
        elif args.filter == 'bocd':
            logp_preds, _ = bocd_filter(pos_set, config)
        elif args.filter == 'vcl':
            logp_preds = vcl_filter(pos_set, config)
        elif args.filter == 'ib':
            logp_preds = ib_filter(pos_set, config)
        else:
            raise NotImplementedError

        meta_logp_preds.append(np.mean(logp_preds))

    print('Average predictive log probability: %.04f +/- %.04f' % 
            (np.mean(meta_logp_preds), np.std(meta_logp_preds)))