import sys
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from plot_lib.plot import PlotTrajectory

def word2id(target, vocab):
    '''
    Argumetns:
    target -- string of target word
    vocab -- list of words

    Returns:
    index of target  word in vocab
    '''
    return vocab.index(target)

def id2word(idx, vocab):
    '''
    Argumetns:
    idx -- index of target  word in vocab
    vocab -- list of words

    Returns:
    string of target word
    '''
    return vocab[idx]

def cos_dist(t, V):
    '''compute cosine distance between a target vector t and all the other word 
    embeddings V. Both t and V are normalized.

    Arguments:
    t -- target vector of shape (L, d) 
    V -- whole vocabulary normalized embeddings of shape (T, L, d) where L is the
        number of words and T is number of timesteps.

    Returns:
    An array of shape (T, L) containing the cosine distance
    '''
    projs = np.sum(V * t, axis=-1)
    return projs

def normalization(vecs_u):
    '''Nomalization of vocabulary embeddings along the second dimension.

    Arguments:
    vecs_u -- shape (L, d)

    Returns:
    A normalized array of shape (L, d) where each column is a unit vector.
    '''
    vecs_u = np.asarray(vecs_u)
    vec_lens = np.sqrt(np.sum(np.square(vecs_u), axis=1))
    vec_lens += 1e-6
    vecs_u = (vecs_u.T / vec_lens).T
    return vecs_u

def sorted_neighbors(t, V):
    '''Sort the indices of embeddings in V based on the cosine distance with t.
    Larger cosine distance has higher priority.

    Arguments:
    t -- target vector of shape (L, d) 
    V -- whole vocabulary normalized embeddings of shape (T, L, d) where L is the
        number of words and T is number of timesteps which is compared for the word.

    Returns:
    An array of shape (T, L) containing the sorted indices for each word. 
    Note the first element should be discarded because it is the word itself.
    '''
    cosine_neigh = cos_dist(t, V)
    idx_sort = np.argsort(-cosine_neigh)
    return idx_sort

def topk_neighbors(word, embedding_ckpt, vocab, topk=5):
    '''Find the Top K neighbors for a given word in a checkpoint file.

    Arguments:
    word -- target word.
    embedding_ckpt -- a numpy array of shape (V, d)
    vocab -- A vocabulary list.
    topk -- the number of neighbors returned. Default is 5.

    Returns:
    A list of words as Top K neighbors for a given word.
    '''
    mat_u = normalization(embedding_ckpt)

    wid = word2id(word, vocab)
    all_neighbors = sorted_neighbors(mat_u[wid, :], mat_u)

    neighbors = []
    for i in range(1, topk + 1):
        neighbors.append(vocab[all_neighbors[i]])
    return neighbors

def get_vocab(vocab_file):
    with open(vocab_file) as file:
        file.readline() # skip first line
        vocab = [line.split('\t')[0] for line in file.readlines()]
    return vocab

def get_trajectory(target_words, embedding_ckpts, vocab, new_axis_set=None):
    '''Given a 1D projected direction, get projections for embeddings on that 
    direction. If `new_axis_set` is not used, the projected direction is the 
    latest word embedding.

    Arguments:
    target_words -- a list of L words that are investigated.
    embedding_ckpt -- a list of T numpy array word embeddings of shape (V, d) or a 
        numpy array of shape (T, V, d).
    vocab -- A vocabulary list.
    new_axis_set -- array of shape (L, d). Default is the latest word embedding.

    Returns:
    An array of shape (T, L).
    '''
    # get embedding vectors
    vecs_set = []
    orig_vecs_set = []
    for mat_u in embedding_ckpts:
        orig_vecs = []
        for target in target_words:
            wid = word2id(target, vocab)
            orig_vecs.append(mat_u[wid, :])
        orig_vecs_set.append(orig_vecs)
        vocab_size = np.minimum(mat_u.shape[0], len(vocab))
        vecs = normalization(orig_vecs)
        vecs_set.append(vecs)

    vecs_set = np.array(vecs_set)
    orig_vecs_set = np.array(orig_vecs_set) # shape (T, len(target_words), d)

    # set the last embedding as the new 1D space
    if not new_axis_set:
        new_axis_set = vecs_set[-1, :, :] # shape (len(target_words), d)
    # new_axis_set = 2 * np.random.random((len(target_words), vecs_set.shape[-1])) - 1
    # new_axis_set = normalization(new_axis_set)
    projects = cos_dist(new_axis_set, vecs_set)

    return projects

def get_neighbors(target_words, embedding_ckpts, vocab, topk=5):
    '''Get neighbors of a list of words from a list of checkpoints.
    '''
    num_timesteps = len(embedding_ckpts)
    diff = int(num_timesteps / 3)
    timestep_idx = list(range(0, num_timesteps, diff))
    word2neighbors = {}
    for word in target_words:
        word2neighbors[word] = []
        for idx in timestep_idx:
            word2neighbors[word].append(topk_neighbors(word, embedding_ckpts[idx], vocab, topk))
    return word2neighbors, timestep_idx

def get_neighbors_by_index(idx, target_words, embedding_ckpts, vocab, topk=5):
    '''Get neighbors of a list of words from a list of checkpoints.
    '''
    timestep_idx = idx
    word2neighbors = {}
    for word in target_words:
        word2neighbors[word] = []
        for idx in timestep_idx:
            word2neighbors[word].append(topk_neighbors(word, embedding_ckpts[idx], vocab, topk))
    return word2neighbors, timestep_idx

def plot_trajectories(target_words, embedding_ckpts, vocab, start_year, end_year, incre, topk=10, tickgap=5):
    '''Plot the trajectory of each word in `target_words` using matplotlib. Each 
    trajectory is saved in separate pdf.

    Time indices of `embedding_ckpts` should meet the ones determined by
    `start_year`, `end_year`, and `incre`.

    Arguments:
    target_words -- a list of L words that are investigated.
    embedding_ckpts -- a list of T numpy array word embeddings of shape (V, d) or a 
        numpy array of shape (T, V, d).
    vocab -- a vocabulary list.
    start_year -- plotting start year of the first timestep.
    end_year -- plotting end year of the last timestep.
    incre -- increment between each timestep.
    topk -- the number of neighbors returned. Default is 10.
    tickgap -- distance between shown ticklabels based on `incre`: 
            `year_diff` = `tickgap` * `incre`
    '''
    trajectories = get_trajectory(target_words, embedding_ckpts, vocab)
    # print(trajectories.shape)
    word2neighbors, timestep_idx = get_neighbors(target_words, embedding_ckpts, vocab, topk=topk)
    # print(timestep_idx)
    year_map = list(range(start_year, end_year + 1, incre))
    for word in target_words:
        trajectory = trajectories[:, target_words.index(word)]
        # print(trajectory)
        canvas = PlotTrajectory(start_year, end_year, incre) #, dpi=96, figsize=(900/96, 500/96))
        canvas.add_title(word)
        canvas.set_style(tickgap=tickgap)
        neighbors = word2neighbors[word]
        canvas.ax.plot(trajectory, lw=1)
        for i in range(len(timestep_idx)):
            if trajectory[timestep_idx[i]] > 0.54:
                canvas.plot_wordset(year_map[timestep_idx[i]], 
                                    neighbors[i], 
                                    offset=trajectory[timestep_idx[i]], 
                                    upwards=False)
            else:
                # print(word + str(year_map[timestep_idx[i]]) + str(trajectory[i]))
                canvas.plot_wordset(year_map[timestep_idx[i]], 
                                    neighbors[i], 
                                    offset=trajectory[timestep_idx[i]], 
                                    upwards=True)
        canvas.save('./' + word + '.pdf')


def main():
    VOCAB_FILE = '../gpu-results-30000-100-le0/vocabs/vocab100000.tsv'
    EMBEDDING_CKPTS = ['../gpu-results-30000-100-le0/training_sess048/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess053/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess058/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess063/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess068/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess073/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess078/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess083/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess088/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess093/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess098/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess103/checkpoint-5000',
                    '../gpu-results-30000-100-le0/training_sess108/checkpoint-5000']
    START_YEAR = 1885
    INCRE = 10 # 2
    END_YEAR = 2005 # 2011 
    YEAR_MAP = list(range(START_YEAR, END_YEAR + 1, INCRE))

    DPI=96

    K_NEIGHBOR = 10
    EPS = 1e-6

    TARGET_WORD = ['atom', 'nitrate', 'parcel', 'space'] # ['actually']# ['clarify', 'atom', 'intelligence'] # 'atom'

    # EMBEDDING_CKPTS = []
    # for i in range(48, 111 + 1):
    #     i = '%03d' % i
    #     EMBEDDING_CKPTS.append('../gpu-results-30000-100-le0/training_sess' + i + '/checkpoint-5000')
    # print(EMBEDDING_CKPTS)

    embs = []
    for ckpt in EMBEDDING_CKPTS:
        reader = tf.train.NewCheckpointReader(ckpt)
        mat_u = reader.get_tensor('q/mean_u')
        embs.append(mat_u)

    vocab = get_vocab(VOCAB_FILE)

    plot_trajectories(TARGET_WORD, embs, vocab, START_YEAR, END_YEAR, INCRE, topk=K_NEIGHBOR)


if __name__ == '__main__':
    # get_trajectory(TARGET_WORD, EMBEDDING_CKPTS, VOCAB_FILE)
    # vocab = get_vocab(VOCAB_FILE)
    # word2neighbors, timestep_idx = get_neighbors(TARGET_WORD, EMBEDDING_CKPTS, vocab)
    # print(word2neighbors)
    # print(topk_neighbors(TARGET_WORD[0], EMBEDDING_CKPTS[0], vocab))
    main()
