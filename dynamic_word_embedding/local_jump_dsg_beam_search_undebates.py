import sys
import os
import shutil
import subprocess
import multiprocessing
from multiprocessing import Pool
import pickle as pkl
import itertools

import numpy as np
from scipy import stats
import tensorflow as tf

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}')
import matplotlib.pyplot as plt

# mute tensorflow information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

if __name__ == '__main__':
    for jump_prior in [-1]: # [-5120, -1280, -320, -40] [-2560, -960, -640] [-480, -240, -160, -120, 0]
        # Data specific variables
        # Options of `(vocab_size, dim)`: (30000, 20), (3000, 10), (10000, 20)
        MAX_VOCAB_SIZE = 30000 
        DIM = 20 #
        DIFFUSION = 4 #

        RAND_VOCAB = False

        BEAM_SIZE = 1
        JUMP_BIAS = jump_prior 

        DATA_FOLDER = "./data-n-ckpt/un-debates/dat"
        IN_DATAPATH_TEMPLATE = DATA_FOLDER + '/npos_%4d.bin.gz'
        VAL_DATAPATH_TEMPLATE = DATA_FOLDER + '/val_npos_%4d.bin.gz'
        CHECKPOINT_FOLDER = (
            './v%d_d%d_b%d_p%d_diff%f_undebates/' % (MAX_VOCAB_SIZE,   
                                                 DIM, 
                                                 BEAM_SIZE, 
                                                 JUMP_BIAS, 
                                                 DIFFUSION))
        OUT_DIR_TEMPLATE = CHECKPOINT_FOLDER + 'undebates%4d_b%s_s%s/' # year, beam_id, s_t
        PREV_PARAM_PATH_TEMPLATE = CHECKPOINT_FOLDER + '%s_%4d.npy' # beam_mu/beam_std, year
        BEAM_VAR_FOLDER = CHECKPOINT_FOLDER + '/beam_results/'
        CONTEXT_VECTOR_CKPT_PATH = (
            './data-n-ckpt/un-debates' 
            '/undebates_entire_timesteps_v%d_d%d/checkpoint-10000' % (MAX_VOCAB_SIZE, DIM))


        FIRST_GPU = 0
        MAX_GPU = 2

        os.mkdir(CHECKPOINT_FOLDER)
        os.mkdir(BEAM_VAR_FOLDER)


        def get_one_hypothesis(diffusion, year, s_t, prev_year, beam_id):
            proc = multiprocessing.current_process()
            proc_id = int(proc.name.split('-')[-1])
            gpu_id = FIRST_GPU + proc_id % MAX_GPU # each process assumes one specific gpu
            
            max_vocab_size = MAX_VOCAB_SIZE
            dim = DIM
            in_datapath = IN_DATAPATH_TEMPLATE % year
            out_dir = OUT_DIR_TEMPLATE % (year, '%d' % beam_id, '%d' % s_t)
            prev_mu = PREV_PARAM_PATH_TEMPLATE % ('beam_mu', prev_year)
            prev_std = PREV_PARAM_PATH_TEMPLATE % ('beam_std', prev_year)
            context_vector_ckpt_path = CONTEXT_VECTOR_CKPT_PATH
            comargs = ['env', 'CUDA_VISIBLE_DEVICES=%d' % gpu_id, 
                       'python', 'local_jump_dsg_singlestep.py', '-E=5000', '--rng_seed=123', '--lr0=0.01', '--lr_exponent=0.0', '--adam_beta2=0.99', '--steps_per_summary=100', '--initial_summaries=100', '--steps_per_checkpoint=1000', 
                       # '--rand_vocab', 
                       # '--norm_npos',
                       '--max_vocab_size=%d' % max_vocab_size, 
                       '-d=%d' % dim, 
                       '--previous_timestep_mu=%s' % prev_mu, 
                       '--previous_timestep_std=%s' % prev_std, 
                       '--beam_id=%d' % beam_id,  
                       '--diffusion=%f' % diffusion,
                       '--mult_diff',
                       '--s_t=%d' % s_t, 
                       '--context_vector_ckpt=%s' % context_vector_ckpt_path,
                       in_datapath, out_dir]

        #     gpu_env = os.environ.copy()
        #     gpu_env['CUDA_VISIBLE_DEVICES'] = ...
            
            if not os.path.exists(out_dir):
                rs = subprocess.run(comargs, check=True, stderr=subprocess.STDOUT)

            elbo = np.load(out_dir + 'elbo.npy')
            assert elbo.shape[0] == max_vocab_size
            return elbo

            
        def get_first_hypothesis(year, s_t):
            proc = multiprocessing.current_process()
            gpu_id = FIRST_GPU
            
            max_vocab_size = MAX_VOCAB_SIZE
            dim = DIM
            in_datapath = IN_DATAPATH_TEMPLATE % year
            beam_id = 0
            out_dir = OUT_DIR_TEMPLATE % (year, '%d' % beam_id, '%d' % s_t)
            context_vector_ckpt_path = CONTEXT_VECTOR_CKPT_PATH
            comargs = ['env', 'CUDA_VISIBLE_DEVICES=%d' % gpu_id, 
                       'python', 'local_jump_dsg_singlestep.py', '-E=5000', '--rng_seed=123', '--lr0=0.01', '--lr_exponent=0.0', '--adam_beta2=0.99', '--steps_per_summary=100', '--steps_per_checkpoint=1000', 
                       # '--rand_vocab', 
                       # '--norm_npos',
                       '--context_vector_ckpt=%s' % context_vector_ckpt_path,
                       '--max_vocab_size=%d' % max_vocab_size, '-d=%d' % dim, 
                       in_datapath, out_dir]
            
            if not os.path.exists(out_dir):
                rs = subprocess.run(comargs, check=True, stderr=subprocess.STDOUT)
            
            elbo = np.load(out_dir + 'elbo.npy')
            assert elbo.shape[0] == max_vocab_size
            return elbo


        def test_procedure(year):
            '''Compute the predictive likelihood for each beam.
            '''
            proc = multiprocessing.current_process()
            gpu_id = FIRST_GPU

            out_file_path = BEAM_VAR_FOLDER + 'likelihood.txt'

            mu_path = PREV_PARAM_PATH_TEMPLATE % ('beam_mu', year)
            log_s_path = BEAM_VAR_FOLDER + 'beam_log_s_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, BEAM_SIZE, year)
            context_vector_ckpt_path = CONTEXT_VECTOR_CKPT_PATH
            val_datapath = VAL_DATAPATH_TEMPLATE % year
            comargs = ['env', 'CUDA_VISIBLE_DEVICES=%d' % gpu_id, 
                       'python', 'dsg_filtering_test.py', 
                       '--rng_seed=123', 
                       # '--rand_vocab', 
                       # '--norm_npos',
                       '--vocab_size=%d' % MAX_VOCAB_SIZE, 
                       '--dim=%d' % DIM, 
                       '--context_vector_ckpt=%s' % context_vector_ckpt_path,
                       '--mu_path=%s' % mu_path,
                       '--log_s_path=%s' % log_s_path,
                       val_datapath, out_file_path]

            rs = subprocess.run(comargs, check=True, stderr=subprocess.STDOUT)

            
        def dsg_beam_search(year_list, s_dim, beam_size=1, jump_bias=-100):
            evidence = [] # store data evidence `elbo_s1 - elbo_s0` for `s_t`

            jump_bias = jump_bias # for random 1000 words # (-2e4 for top 1000 words)
            diffusion = DIFFUSION

            beam_log_s = np.zeros((1, s_dim, 1)) # (beam_size, s_dim, T)
            beam_decision = np.ones((1, s_dim, 1))
            emb_dim = DIM
            beam_mu = np.ones((1, s_dim, emb_dim, 1), dtype=np.float32) # (beam_size, s_dim, emb_dim, 1)
            beam_std = np.ones((1, s_dim, emb_dim, 1), dtype=np.float32) # (beam_size, s_dim, emb_dim, 1)

            history_decisions, history_log_s = [], []

            log_num_beams = 1e-6 # add an epsilon to keep stability

            start_year = year_list[0] # 43
            end_year = year_list[-1]# 111
            for (i, year) in enumerate(year_list):
                if year == start_year:
                    with Pool(1) as p:
                        elbos = p.starmap(get_first_hypothesis, [(year, 1)])
                        p.close()
                    checkpoint_path = tf.train.latest_checkpoint(OUT_DIR_TEMPLATE % (year, '%d' % 0, '%d' % 1))
                    reader = tf.train.NewCheckpointReader(checkpoint_path)
                    mean, std = reader.get_tensor('q/mean_u'), np.exp(reader.get_tensor('q/log_std_u'))
                    beam_mu *= np.expand_dims(mean, axis=-1)
                    beam_std *= np.expand_dims(std, axis=-1)
                else:
                    prev_year = year_list[i - 1]
                    log_num_beams += np.log(2) # log(number of hypothesis until the end of this step)
                    if log_num_beams >= np.log(beam_size * 2):
                        num_effect_beams = beam_size
                    else:
                        num_effect_beams = int(np.exp(log_num_beams - np.log(2)))

                    diffusions = [diffusion] * num_effect_beams * 2
                    years = [year] * num_effect_beams * 2
                    s_ts = [1] * num_effect_beams + [0] * num_effect_beams
                    prev_years = [prev_year] * num_effect_beams * 2
                    beam_ids = list(range(num_effect_beams)) * 2

                    candidate_args = zip(diffusions, years, s_ts, prev_years, beam_ids)
                    with Pool(np.min((num_effect_beams * 2, MAX_GPU))) as p:
                        elbos = p.starmap(get_one_hypothesis, candidate_args)
                        p.close()
                    elbo_s1 = np.array(elbos[:num_effect_beams]) # shape (num_effect_beams, s_dim)
                    elbo_s0 = np.array(elbos[num_effect_beams:]) # shape (num_effect_beams, s_dim)
                    
                    print(elbo_s1, elbo_s0, elbo_s1 - elbo_s0)
                    evidence.append(elbo_s1 - elbo_s0)
                    # out_dir = OUT_DIR_TEMPLATE % (start_year, '%d' % 0, '%d' % 1)
                    # dat_sum = np.load(out_dir + 'dat_sum.npy')
                    dat_sum = 1.0
                    elbo_s1 /= dat_sum
                    elbo_s0 /= dat_sum
                    tempered_jump_bias = jump_bias/dat_sum
                    q_log_s1 = -np.log1p(np.exp(-(elbo_s1 - elbo_s0 + tempered_jump_bias))) # shape (beam_size, s_dim)
                    q_log_s0 = -np.log1p(np.exp(elbo_s1 - elbo_s0 + tempered_jump_bias)) 
                    
                    # process the `-inf` values by identity
                    # `q_log_s1 - q_log_s0 = elbo_s1 - elbo_s0 + jump_bias`
                    # and the fact that one of `q_log_s1` and `q_log_s0` should be zero
                    inf_ind = np.argwhere(q_log_s1 == -np.inf)
                    q_log_s1[inf_ind[:, 0], inf_ind[:, 1]] = (elbo_s1 - elbo_s0 + tempered_jump_bias)[inf_ind[:, 0], inf_ind[:, 1]]
                    inf_ind = np.argwhere(q_log_s0 == -np.inf)
                    q_log_s0[inf_ind[:, 0], inf_ind[:, 1]] = -(elbo_s1 - elbo_s0 + tempered_jump_bias)[inf_ind[:, 0], inf_ind[:, 1]]
                    assert np.all(q_log_s1 <= 0) and np.all(q_log_s0 <= 0)
                    assert not (np.any(q_log_s1 == -np.inf) or np.any(q_log_s0 == -np.inf))
                    
                    # beam search
                    # TODO: simplify the wordy code
                    print('Beam search for year %d...' % year)
                    sys.stdout.flush()
                    candidate_beam_log_s = np.tile(beam_log_s[:, :, -1], (2, 1)) + np.squeeze(np.concatenate((q_log_s1, q_log_s0), axis=0)) # shape (2*beam_size, s_dim)
                    # beam diversification
                    ind = []
                    for scores in candidate_beam_log_s.transpose():
                        ind.append(find_optimal_beam(scores, beam_size))
                    ind = np.transpose(ind)
                    print("Shape of selection indices:", np.shape(ind))
                    sys.stdout.flush()
                    # ind = np.argsort(-candidate_beam_log_s, axis=0)[:beam_size, :] # (beam_size, s_dim)
                    extended_beam_log_s = np.tile(beam_log_s, (2, 1, 1))
                    _beam_log_s = []
                    for (extended_beam_log_s_i, candidate_beam_log_s_i, ind_i) in zip(np.transpose(extended_beam_log_s, (1, 0, 2)), np.transpose(candidate_beam_log_s, (1, 0)), np.transpose(ind, (1, 0))):
                        _beam_log_s.append(np.expand_dims(np.concatenate((extended_beam_log_s_i[ind_i, :], np.expand_dims(candidate_beam_log_s_i[ind_i], 1)), axis=-1), 1))
                    beam_log_s = np.hstack(_beam_log_s) # (beam_size, s_dim, T)
                    # print(beam_log_s.shape, beam_log_s)
                    
                    extended_candidate_decision = np.expand_dims(np.repeat([1, 0], num_effect_beams), 1) # e.g. [1, 1, 0, 0]
                    extended_beam_decision = np.tile(beam_decision, (2, 1, 1)) # e.g. [0, 1, 0, 1]
                    _beam_decision = []
                    for (extended_beam_decision_i, ind_i) in zip(np.transpose(extended_beam_decision, (1, 0, 2)), np.transpose(ind, (1, 0))):
                        _beam_decision.append(np.expand_dims(np.concatenate((extended_beam_decision_i[ind_i, :], extended_candidate_decision[ind_i]), axis=-1), 1))
                    beam_decision = np.hstack(_beam_decision) # (beam_size, s_dim, T)
                    print(beam_decision[0, 0, :])
                    sys.stdout.flush()

                    # beam search for embeddings
                    #beam_emb = ... # (beam_size, s_dim, emb_dim, T-1)
                    dir_0 = OUT_DIR_TEMPLATE % (year, '%d', '0')
                    dir_1 = OUT_DIR_TEMPLATE % (year, '%d', '1')
                    beam_mu, beam_std = update_mu_std(ind, dir_1, dir_0, beam_mu, beam_std, num_effect_beams)

                    # delete unused checkpoint folder
                    for b in range(num_effect_beams):
                        if os.path.exists(dir_1 % b) and os.path.isdir(dir_1 % b):
                            shutil.rmtree(dir_1 % b)
                            print('Remove directory %s' % (dir_1 % b))
                        if os.path.exists(dir_0 % b) and os.path.isdir(dir_0 % b):
                            shutil.rmtree(dir_0 % b)
                            print('Remove directory %s' % (dir_0 % b))

                    # add to history lists
                    # history_decisions.append(beam_decision)
                    # history_log_s.append(beam_log_s)

                np.save(PREV_PARAM_PATH_TEMPLATE % ('beam_mu', year), beam_mu)
                np.save(PREV_PARAM_PATH_TEMPLATE % ('beam_std', year), beam_std)

                # np.save(BEAM_VAR_FOLDER + 'evidence_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, beam_size, year), np.array(evidence))
                np.save(BEAM_VAR_FOLDER + 'beam_decision_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, beam_size, year), beam_decision)
                np.save(BEAM_VAR_FOLDER + 'beam_log_s_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, beam_size, year), beam_log_s)

                print('Saving beam search results done.')

                # delete outdated saved parameters
                prev_year = year - 1
                rm_file_path = [PREV_PARAM_PATH_TEMPLATE % ('beam_mu', prev_year), 
                    PREV_PARAM_PATH_TEMPLATE % ('beam_std', prev_year), 
                    BEAM_VAR_FOLDER + 'beam_decision_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, beam_size, prev_year),
                    BEAM_VAR_FOLDER + 'beam_log_s_v%d_d%d_b%d_y%d.npy' % (MAX_VOCAB_SIZE, DIM, beam_size, prev_year)]
                for fpath in rm_file_path:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        print('Remove:', fpath)

                # pkl.dump(history_log_s, open(BEAM_VAR_FOLDER + 'history_log_s_v%d_d%d_b%d.pkl' % (MAX_VOCAB_SIZE, DIM, beam_size), 'wb'))
                # pkl.dump(history_decisions, open(BEAM_VAR_FOLDER + 'history_decisions_v%d_d%d_b%d.pkl' % (MAX_VOCAB_SIZE, DIM, beam_size), 'wb'))

                # test procedure
                print('Computing predictive likelihood for heldout data.')
                test_args = [[year]]
                with Pool(1) as p:
                    p.starmap(test_procedure, test_args)
                    p.close()
                    
                print('Year %d done.' % year)
                sys.stdout.flush()

            return beam_decision, beam_log_s

        def find_optimal_beam(scores, beam_size, discard_fraction = 1.0 / 3.0):
            '''Return the indices of the `beam_size` optimal hypotheses.

            Args:
                scores: vector of scores (e.g., log probabilities or ELBOs) of each
                    hypothesis. Must have an even length and the two hypotheses with the
                    same parent always have to come together, i.e.,
                    scores = [
                        score of the first child of the first parent,
                        score of the first child of the second parent,
                        score of the first child of the third parent,
                        score of the second child of the first parent,
                        score of the second child of the second parent,
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
            parents = np.array(list(range(num_parents)) * 2).flatten()
            
            # Discard `discard_fraction` of the hypotheses (except that we don't have to discard
            # any hypothesis in the first few steps when there are only few hypotheses)
            # num_keep = min(len(scores), round((1.0 - discard_fraction) * (2 * beam_size)))
            num_keep = len(scores) - 2
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


        def update_mu_std(ind, dir_1, dir_0, beam_mu, beam_std, beam_size):
            """dir_1 = ... # template with beam id as indicator
            dir_0 = ...
            """
            mu_1, std_1 = [], []
            mu_0, std_0 = [], []
            for b in range(beam_size):
                checkpoint_path = tf.train.latest_checkpoint(dir_1 % b)
                reader = tf.train.NewCheckpointReader(checkpoint_path)
                _mean_1, _std_1 = reader.get_tensor('q/mean_u'), np.exp(reader.get_tensor('q/log_std_u'))
                mu_1.append(_mean_1)
                std_1.append(_std_1)
                checkpoint_path = tf.train.latest_checkpoint(dir_0 % b)
                reader = tf.train.NewCheckpointReader(checkpoint_path)
                _mean_0, _std_0 = reader.get_tensor('q/mean_u'), np.exp(reader.get_tensor('q/log_std_u'))
                mu_0.append(_mean_0)
                std_0.append(_std_0)

            mu_1, std_1 = np.asarray(mu_1, dtype=np.float32), np.asarray(std_1, dtype=np.float32) # (beam_size, vocab_size, emb_size)
            mu_0, std_0 = np.asarray(mu_0, dtype=np.float32), np.asarray(std_0, dtype=np.float32)

            extended_candidate_mu = np.concatenate((mu_1, mu_0), axis=0) # (2*beam_size, vocab_size, emb_size)
            extended_beam_mu = np.tile(beam_mu, (2, 1, 1, 1)) # (2*beam_size, vocab_size, emb_size, T-1)
            # process for each word
            _beam_mu = []
            for (extended_beam_mu_i, extended_candidate_mu_i, ind_i) in zip(extended_beam_mu.transpose(1, 0, 2, 3), extended_candidate_mu.transpose(1, 0, 2), ind.transpose()):
                _beam_mu.append(np.expand_dims(np.concatenate((extended_beam_mu_i[ind_i, :, :], np.expand_dims(extended_candidate_mu_i[ind_i, :], axis=-1)), axis=-1), axis=1))
            beam_mu = np.concatenate(_beam_mu, axis=1)

            extended_candidate_std = np.concatenate((std_1, std_0), axis=0) # (2*beam_size, vocab_size, emb_size)
            extended_beam_std = np.tile(beam_std, (2, 1, 1, 1)) # (2*beam_size, vocab_size, emb_size, T-1)
            # process for each word
            _beam_std = []
            for (extended_beam_std_i, extended_candidate_std_i, ind_i) in zip(extended_beam_std.transpose(1, 0, 2, 3), extended_candidate_std.transpose(1, 0, 2), ind.transpose()):
                _beam_std.append(np.expand_dims(np.concatenate((extended_beam_std_i[ind_i, :, :], np.expand_dims(extended_candidate_std_i[ind_i, :], axis=-1)), axis=-1), axis=1))
            beam_std = np.concatenate(_beam_std, axis=1)

            return beam_mu, beam_std


        year_list = range(1970, 2018 + 1)
        # Training
        _, _ = dsg_beam_search(year_list, s_dim=MAX_VOCAB_SIZE, beam_size=BEAM_SIZE, jump_bias=JUMP_BIAS)
        
        # Test procedure
        # for year in year_list:
        #     print('Computing predictive likelihood for heldout data.')
        #     test_args = [[year]]
        #     with Pool(1) as p:
        #         p.starmap(test_procedure, test_args)
        #         p.close()
        #     print('Year %d done.' % year)
        #     sys.stdout.flush()


# tools
def delayed_update_procedure(beam_decision, beam_mu, num_years, update=True, beam_id=0, vocab_size=30000, emb_dim=100):
    '''`beam_decision` should be a numpy ndarray of shape 
        `(beam_size, vocab_size, num_steps)`.

        `beam_mu` is the inferred embeddings of shape 
        `(beam_size, vocab_size, emb_dim, num_steps)`.

        `num_years` is `num_steps`.

        It returns word embeddings of shape `(num_steps, vocab_size, emb_dim)`.
    '''
    embs = beam_mu[beam_id, ...].transpose(0, 2, 1)
    if update:
        embeddings = np.zeros((vocab_size, num_years, emb_dim))
        # process each word
        for i in range(vocab_size):
            beam_decision_i = beam_decision[beam_id, i, :]
            stay_start = 0
            for j, d_i in enumerate(beam_decision_i):
                if d_i == 1 and j > 0:
                    embeddings[i, stay_start:j, :] = np.tile(embs[i, j-1, :], 
                                                             (j - stay_start, 1))
                    stay_start = j
            embeddings[i, stay_start:, :] = np.tile(embs[i, -1, :], 
                                                    (num_years - stay_start, 1))
        embeddings = np.transpose(embeddings, (1, 0, 2))
    else:
        embeddings = embs

    return embeddings
