import sys
import numpy as np
import tensorflow as tf

from plot_lib.plot import PlotMeaningShift

#VOCAB_FILE = './data/npos_vocabsize100000/vocab_1800to2008_vocabsize100000.tsv'
VOCAB_FILE = '../gpu-results-30000-100-le0/vocabs/vocab100000.tsv'
# EMBEDDING_CKPT = './training_year1960/checkpoint-10000'
#EMBEDDING_CKPT = './training_year2007/checkpoint-5000'
EMBEDDING_CKPTS = ['../gpu-results-30000-100-le0/training_sess045/checkpoint-5000',
                '../gpu-results-30000-100-le0/training_sess091/checkpoint-5000']
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
K_NEIGHBOR = 6

EPS = 1e-6

THRESHOLD_1 = 2.0
THRESHOLD_2 = 2.0

TOP_N = 5

class WordRules:
    def __init__(self, non_digit=True, non_capital=True, non_special=True):
        self.non_special = non_digit
        self.non_capital = non_capital
        self.non_special = non_special

    def filter(self, word):
        for c in word:
            if self.non_special:
                if c.isdigit():
                    return True
            if self.non_capital:
                if c.isupper():
                    return True
            if self.non_special:
                if not c.isalpha():
                    return True
        return False

def nearest_id(dist):
    # note that there is NaN, so reverse the order
    idx_sort = np.argsort(-dist)
    return idx_sort[0]

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
    embeddings V.

    Arguments:
    t -- target vector of shape (d,) 
    V -- whole vocabulary normalized embeddings of shape (L, d) where L is the
        number of words.

    Returns:
    A vector of length L containing the cosine distance
    '''
    # t_len = np.sqrt(np.sum(np.square(t)))
    # t /= t_len
    dot_prods = np.matmul(V, t)
    return dot_prods

def normalization(vecs_u):
    '''Nomalization of vocabulary embeddings

    Arguments:
    vecs_u -- shape (L, d)
    '''
    vec_lens = np.sqrt(np.sum(np.square(vecs_u), axis=1))
    vec_lens += 1e-6
    vecs_u = (vecs_u.T / vec_lens).T
    return vecs_u

def point_cos_dist(u, v):
    '''u and v are unit vectors.
    '''
    return np.dot(u, v)

def global_measure(U, V):
    '''Implement global measure in paper ``Cultural Shift or Linguistic Drift? 
    Comparing Two Computational Measures of Semantic Change''

    Arguments:
    U and V are matrices of shape (L, d).

    Returns:
    An array of length L.
    '''
    dist = []
    for i in range(np.shape(U)[0]):
        dist.append(point_cos_dist(U[i, :], V[i, :]))
    return np.array(dist)

def find_neighbors(t ,V):
    cosine_neigh = cos_dist(t, V)
    idx_sort = np.argsort(-cosine_neigh)
    return idx_sort

def veclen(t):
    return np.sqrt(np.sum(np.square(t)))

def find_change_words(embedding1, embedding2, vocab, thres_1, thres_2, topk=5, upper_thres_1=10):
    '''Find Top K words with the most drastic change between two timesteps, 
    `embedding1` and `embedding2`.

    Some rules are exerted when selecting the words:
    - Words that have capital letters, numbers, and special characters are not contained.
    - Words whose previous embeddings are shorter than `thres_1` or second
      embeddings are shorter than `thres_2` are not contained.
    - Words whose recent embeddings are longer than `upper_thres_1` are not 
      contained.

    Arguments:
    embedding1 -- embeddings for the first timestep of shape (V, d).
    embedding2 -- embeddings for the second timestep of shape (V, d).
    vocab -- vocabulary list
    thres_1 -- lower threshold for previous embeddings
    thres_2 -- lower threshold for recent embeddings
    upper_thres_1 -- upper threshold for previous embeddings. Defualt is 10.0
    topk -- the number of words that are returned.

    Returns:
    A list of `topk` words that have the drastic change between the two given 
    timesteps.
    '''
    # get vectors
    vecs = [] 
    orig_vecs = [embedding1, embedding2]
    for emb in orig_vecs:
        vocab_size = np.minimum(emb.shape[0], len(vocab))
        vecs_u = normalization(emb)
        vecs.append(vecs_u)

    # print('Vocabulary size: %d ("%s", "%s", ..., "%s", "%s")' 
    #     % (vocab_size, vocab[0], vocab[1], vocab[vocab_size - 2], vocab[vocab_size - 1]))

    # add word rules
    wordrule = WordRules()

    # global measures
    global_dist = global_measure(vecs[0], vecs[1])
    idx_sort = np.argsort(global_dist)
    top_n = topk
    # print('TOKEN\tCOSINE\tLENGTH1\tLENGTH2')
    i = -1
    targets = []
    while top_n != 0:
        i += 1
        cur_idx = idx_sort[i]
        length1 = veclen(orig_vecs[0][cur_idx, :])
        length2 = veclen(orig_vecs[1][cur_idx, :])
        if global_dist[cur_idx] == 0: 
            if length1 == 0 and length2 == 0:
                continue
        # print distance and length
        if length1 < thres_1 or length1 > upper_thres_1 or length2 < thres_2:
            continue
        if wordrule.filter(vocab[cur_idx]):
            continue
        print('%s\t%.3f\t%.3f\t%.3f' % (vocab[cur_idx], global_dist[cur_idx], length1, length2))
        targets.append(vocab[cur_idx])
        top_n -= 1
        # print neighbors
        neighbors_prev = find_neighbors(vecs[0][cur_idx, :], vecs[0])
        neighbors_recent = find_neighbors(vecs[1][cur_idx, :], vecs[1])
        for j in range(1, topk + 1):
            print('%s:\t%s\t%s' % (vocab[cur_idx], vocab[neighbors_prev[j]], vocab[neighbors_recent[j]]))

    return targets

def plot_meaning_shift(embedding_ckpts, vocab_path, start_year, end_year, incre):
    with open(vocab_path) as file:
        file.readline() # skip first line
        vocab = [line.split('\t')[0] for line in file.readlines()]

    canvas = PlotMeaningShift(start_year, end_year, incre)
    canvas.set_style()

    year = start_year

    for (ckpt1, ckpt2) in zip(embedding_ckpts[:-1], embedding_ckpts[1:]):
        reader = tf.train.NewCheckpointReader(ckpt1)
        emb1 = reader.get_tensor('q/mean_u')
        reader = tf.train.NewCheckpointReader(ckpt2)
        emb2 = reader.get_tensor('q/mean_u')
        print('----------------------')
        print(str(year) + '-' + str(year + incre))
        print('----------------------')
        print('FREQUENT WORDS THAT CHANGED THE MOST')
        thres_1 = 2.0
        thres_2 = 2.0
        words = find_change_words(emb1, emb2, vocab, thres_1, thres_2)
        canvas.plot_wordset(year, words, offset=1, tot_group=2)
        print('WORDS THAT BEGAN TO BE USED')
        thres_1 = 0.0
        thres_2 = 2.0
        words = find_change_words(emb1, emb2, vocab, thres_1, thres_2, upper_thres_1=1.0)
        canvas.plot_wordset(year, words, offset=0, tot_group=2, color='blue')
        year += incre

    canvas.plot_legends()

    # canvas.show()
    canvas.save('./meanings_shift.pdf')

def main():

    START_YEAR = 1885
    END_YEAR = 2005
    INCRE = 10
    DPI=96

    plot_meaning_shift(EMBEDDING_CKPTS, VOCAB_FILE, START_YEAR, END_YEAR, INCRE)

    # canvas = PlotMeaningShift(START_YEAR, END_YEAR, INCRE, dpi=DPI)
    # canvas.set_style()

    # year = START_YEAR

    # for (ckpt1, ckpt2) in zip(EMBEDDING_CKPTS[:-1], EMBEDDING_CKPTS[1:]):
    #     print('----------------------')
    #     print(str(year) + '-' + str(year + INCRE))
    #     print('----------------------')
    #     print('FREQUENT WORDS THAT CHANGED THE MOST')
    #     thres_1 = 2.0
    #     thres_2 = 2.0
    #     words = find_change_words(ckpt1, ckpt2, VOCAB_FILE, thres_1, thres_2)
    #     canvas.plot_wordset(year, words, offset=1, tot_group=2)
    #     print('WORDS THAT BEGAN TO BE USED')
    #     thres_1 = 0.0
    #     thres_2 = 2.0
    #     words = find_change_words(ckpt1, ckpt2, VOCAB_FILE, thres_1, thres_2, upper_thres_1=1.0)
    #     canvas.plot_wordset(year, words, offset=0, tot_group=2, color='blue')
    #     year += INCRE

    # canvas.plot_legends()

    # # canvas.show()
    # canvas.save('./meanings_shift.pdf')

def test():
    a = np.array([[1, 2]])
    b = np.array([[3, 4]])
    a = normalization(a)
    b = normalization(b)
    print(a, b)
    print(point_cos_dist(np.squeeze(a), np.squeeze(b)))


if __name__ == '__main__':
    main()
