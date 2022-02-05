import os
import sys
import numpy as np
import argparse

from filters import vbs_filter, bocd_filter, bf_filter, vcl_filter, ib_filter
from dataloader import get_malware_dataset
from dataloader import get_elec2_dataset
from dataloader import get_sensordrift_dataset


# datasets = ['malware', 'elec2', 'sensordrift']
# validations = [True, False]
# methods = ['vbs', 'bocd', 'bf', 'vcl', 'ib']


configs = {
    'malware': {
        'dataset': 'malware',
        'logodds': True,
        'num_feature': 483,
        'sigma_n': 40,
        'bocd': {
            'K': 6, # ensemble size 
            'hazard': 0.3,
        },
        'bf': {
            'beta': 0.999,
        },
        'vbs': {
            'K': 1, # ensemble size
            'beta': {
                1: 1.2, # 1/"beta" in the sense of bf method
                3: 1.07,
                6: 1.05,
            }, 
            'p': {
                1: 0.5,
                3: 0.5,
                6: 0.5,
            },
        },
    },

    'elec2': {
        'dataset': 'elec2',
        'logodds': True,
        'num_feature': 15,
        'sigma_n': 0.01,
        'bocd': {
            'K': 6, # ensemble size 
            'hazard': 0.9,
        },
        'bf': {
            'beta': 0.98,
        },
        'vbs': {
            'K': 1, # ensemble size
            'beta': {
                1: 1.2, # 1/"beta" in the sense of bf method
                3: 1.2,
                6: 1.2,
            }, 
            'p': {
                1: 0.5,
                3: 0.5,
                6: 0.5,
            },
        },
    },

    'sensordrift': {
        'dataset': 'sensordrift',
        'logodds': False,
        'num_feature': 129,
        'sigma_n': 1.,
        'bocd': {
            'K': 6, # ensemble size 
            'hazard': 0.2,
        },
        'bf': {
            'beta': 0.9,
        },
        'vbs': {
            'K': 1, # ensemble size
            'beta': {
                1: 0.7, 
                3: 0.7,
                6: 0.7,
            }, 
            'p': {
                1: 0.507,
                3: 0.507,
                6: 0.507,
            },
        },
    }
}


def cum_average_l1_error(abs_errs):
    return np.cumsum(abs_errs) / np.linspace(1, len(abs_errs), len(abs_errs))


parser = argparse.ArgumentParser(description='Experiment configurations.')
parser.add_argument('--dataset', type=str, default='elec2')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--filter', type=str, default='vbs')


if __name__ == "__main__":

    args = parser.parse_args()

    if args.dataset == 'sensordrift':
        config = configs['sensordrift']
        x_train_set, y_train_set, x_test_set, y_test_set = get_sensordrift_dataset(valid=args.validation)
    
    elif args.dataset == 'malware':
        config = configs['malware']
        x_train_set, y_train_set, x_test_set, y_test_set = get_malware_dataset(valid=args.validation)

    elif args.dataset == 'elec2':
        config = configs['elec2']
        x_train_set, y_train_set, x_test_set, y_test_set = get_elec2_dataset(valid=args.validation)

    else:
        raise NotImplementedError

    if args.filter == 'vbs':
        abs_errs, _ = vbs_filter(x_train_set, y_train_set, x_test_set, y_test_set, config)
    elif args.filter == 'bf':
        abs_errs = bf_filter(x_train_set, y_train_set, x_test_set, y_test_set, config)
    elif args.filter == 'bocd':
        abs_errs, _ = bocd_filter(x_train_set, y_train_set, x_test_set, y_test_set, config)
    elif args.filter == 'vcl':
        abs_errs = vcl_filter(x_train_set, y_train_set, x_test_set, y_test_set, config)
    elif args.filter == 'ib':
        abs_errs = ib_filter(x_train_set, y_train_set, x_test_set, y_test_set, config)
    else:
        raise NotImplementedError

    cum_ave = cum_average_l1_error(abs_errs)
    print('MCAE: %.04f' % (np.mean(cum_ave)))