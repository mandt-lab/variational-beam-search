'''TODO: document
'''

import struct
import sys
import gzip
import numpy as np
from scipy import sparse

# TODO: I'd rather not depend on internal library functions, but I don't see a way around this.
from scipy.sparse._sparsetools import csr_sample_offsets


class FactorizedSingleTimestep:
    '''Word-context co-occurrence matrix for a single time step.'''

    def __init__(self, path, vocab_size, u_vocab_ids, rng=None, dense=False, neg_ratio=1.0, neg_exp=0.75, norm_npos=False):
        '''Read a co-occurrence matrix from a file.

        Expects a gzip compressed binary file containing the following
        numbers concatenated in this order in little endian byte order:
        - `vocab_size`: one unsigned 4-byte integer
        - `nnz` (i.e., the number of non-zero entries): one unsigned 4-byte integer
        - `indptr` (used for sparse CSR matrix): `vocab_size + 1` unsigned 4-byte integers
        - `indices` (used for sparse CSR matrix): `nnz` unsigned 4-byte integers
        - `data` (i.e., the nonzero values): `nnz` single precision floats
        - end of file.
        '''
        with gzip.open(path, 'rb') as file:
            header_bytes = file.read(2*4)
            orig_vocab_size, nnz = struct.unpack("<2L", header_bytes)
            indptr = np.frombuffer(
                file.read(4 * (orig_vocab_size + 1)), dtype=np.int32)
            indices = np.frombuffer(file.read(4 * nnz), dtype=np.int32)
            data = np.frombuffer(file.read(4 * nnz), dtype=np.float32)
            assert len(file.read(1)) == 0  # Assert EOF.

        if sys.byteorder == 'big':
            # The file is written in little endian, so we need to byteswap.
            indptr.byteswap(inplace=True)
            indices.byteswap(inplace=True)
            data.byteswap(inplace=True)

        assert indptr[-1] == nnz
        n_pos = sparse.csr_matrix(
            (data, indices, indptr), shape=(orig_vocab_size, orig_vocab_size))

        if vocab_size < 0:
            vocab_size = orig_vocab_size
        else:
            assert orig_vocab_size >= vocab_size
            if rng is None:
                # take top `vocab_size`
                n_pos = n_pos[:vocab_size, :vocab_size]
            else:
                # take random `vocab_size`
                vocab_indices = np.sort(rng.choice(30000, vocab_size, replace=False))
                n_pos = n_pos[vocab_indices, :]
                n_pos = n_pos[:, vocab_indices]

        u_vocab_size = len(u_vocab_ids)
        if u_vocab_size == 0:
            u_vocab_size = vocab_size
        else:
            assert max(u_vocab_ids) <= vocab_size
            u_n_pos = n_pos[u_vocab_ids, :]

        self.counts_pos = np.squeeze(np.asarray(u_n_pos.sum(axis=1)))
        tot_counts_pos = np.squeeze(np.asarray(n_pos.sum(axis=1)))
        if neg_ratio is not None:
            self.scaled_freq_neg = np.power(tot_counts_pos, neg_exp)
            self.scaled_freq_neg *= neg_ratio / self.scaled_freq_neg.sum()

        self.time_stamp = None  # TODO
        self.vocab_size = vocab_size
        self.u_vocab_size = u_vocab_size

        if dense:
            # Calculate `n_sum = n^+ + n^-`. This is what the model ultimately uses.
            self.n_sum = np.asarray(u_n_pos.todense() +
                          np.outer(self.counts_pos, self.scaled_freq_neg))  # shape (V_u, V)
        else:
            self.n_pos = n_pos



class SingleTimestep:
    '''Word-context co-occurrence matrix for a single time step.'''

    def __init__(self, path, vocab_size, rng=None, dense=False, neg_ratio=1.0, neg_exp=0.75, norm_npos=False):
        '''Read a co-occurrence matrix from a file.

        Expects a gzip compressed binary file containing the following
        numbers concatenated in this order in little endian byte order:
        - `vocab_size`: one unsigned 4-byte integer
        - `nnz` (i.e., the number of non-zero entries): one unsigned 4-byte integer
        - `indptr` (used for sparse CSR matrix): `vocab_size + 1` unsigned 4-byte integers
        - `indices` (used for sparse CSR matrix): `nnz` unsigned 4-byte integers
        - `data` (i.e., the nonzero values): `nnz` single precision floats
        - end of file.
        '''
        with gzip.open(path, 'rb') as file:
            header_bytes = file.read(2*4)
            orig_vocab_size, nnz = struct.unpack("<2L", header_bytes)
            indptr = np.frombuffer(
                file.read(4 * (orig_vocab_size + 1)), dtype=np.int32)
            indices = np.frombuffer(file.read(4 * nnz), dtype=np.int32)
            data = np.frombuffer(file.read(4 * nnz), dtype=np.float32)
            assert len(file.read(1)) == 0  # Assert EOF.

        if sys.byteorder == 'big':
            # The file is written in little endian, so we need to byteswap.
            indptr.byteswap(inplace=True)
            indices.byteswap(inplace=True)
            data.byteswap(inplace=True)

        assert indptr[-1] == nnz
        n_pos = sparse.csr_matrix(
            (data, indices, indptr), shape=(orig_vocab_size, orig_vocab_size))

        if vocab_size < 0:
            vocab_size = orig_vocab_size
        else:
            assert orig_vocab_size >= vocab_size
            if rng is None:
                # take top `vocab_size`
                n_pos = n_pos[:vocab_size, :vocab_size]
            else:
                # take random `vocab_size`
                vocab_indices = np.sort(rng.choice(30000, vocab_size, replace=False))
                n_pos = n_pos[vocab_indices, :]
                n_pos = n_pos[:, vocab_indices]

        self.counts_pos = np.squeeze(np.asarray(n_pos.sum(axis=1)))
        if neg_ratio is not None:
            self.scaled_freq_neg = np.power(self.counts_pos, neg_exp)
            self.scaled_freq_neg *= neg_ratio / self.scaled_freq_neg.sum()

        self.time_stamp = None  # TODO
        self.vocab_size = vocab_size

        if norm_npos:
            # normalize
            n_pos /= n_pos.sum()
            # recompute
            self.counts_pos = np.squeeze(np.asarray(n_pos.sum(axis=1)))

        if dense:
            # Calculate `n_sum = n^+ + n^-`. This is what the model ultimately uses.
            self.n_sum = np.asarray(n_pos.todense() +
                          np.outer(self.counts_pos, self.scaled_freq_neg))  # shape (V, V)
        else:
            self.n_pos = n_pos


class MultipleTimesteps:
    '''Word-context co-occurrence tensor over multiple time steps.'''

    def __init__(self, path, neg_ratio=1.0, neg_exp=0.75):
        '''Read a co-occurrence tensor from a file.

        Expects a gzip compressed binary file containing the following
        numbers concatenated in this order in little endian byte order:
        - `vocab_size`: unsigned 4-byte integer
        - `num_timesteps`: unsigned 4-byte integer
        - `num_ij` (i.e., the number of different word-context pairs that appear in at least
          one time step): unsigned 4-byte integer
        - `nnz` (i.e., the total number of non-zero entries of the rank-3 tensor):
          unsigned 4-byte integer
        - `counts_pos`: word counts, flattened `float32` matrix of shape
          `(vocab_size, num_timesteps)`
        - A CSR matrix of shape `(vocab_size, vocab_size)`, containing the indices into the 
          timesamps and data fields, stored as follows:
            - `indptr`: `vocab_size + 1` signed 4-byte integers
            - `indices`: `nnz` signed 4-byte integers
            - `indptr2`: `num_ij` signed 4-byte integers
        - The timestamps of all data points: `nnz` signed 4-byte integers.
        - The co-occurrence counts: `nnz` `float32` values.
        - end of file.
        '''
        with gzip.open(path, 'rb') as file:
            header_bytes = file.read(4 * 8)
            self.vocab_size, self.num_timesteps, num_ij, nnz = struct.unpack(
                "<4Q", header_bytes)
            self.counts_pos = np.frombuffer(
                file.read(4 * self.vocab_size * self.num_timesteps), dtype=np.float32).reshape(
                    (self.vocab_size, self.num_timesteps))
            self.indptr = np.frombuffer(
                file.read(4 * (self.vocab_size + 1)), dtype=np.int32)
            self.indices = np.frombuffer(
                file.read(4 * num_ij), dtype=np.int32)
            self.indptr2 = np.frombuffer(
                file.read(8 * (num_ij + 1)), dtype=np.int64)
            self.timesteps = np.frombuffer(file.read(2 * nnz), dtype=np.int16)
            self.data = np.frombuffer(file.read(4 * nnz), dtype=np.float32)
            assert self.data.shape == (nnz,)
#            assert len(file.read(1)) == 0  # Assert EOF.

        if sys.byteorder == 'big':
            # The file is written in little endian, so we need to byteswap.
            self.counts_pos.byteswap(inplace=True)
            self.indptr.byteswap(inplace=True)
            self.indices.byteswap(inplace=True)
            self.indptr2.byteswap(inplace=True)
            self.timesteps.byteswap(inplace=True)
            self.data.byteswap(inplace=True)

        assert self.indptr[-1] == num_ij
        assert self.indptr2[-1] == nnz

        if neg_ratio is not None:
            self.scaled_freq_neg = np.power(self.counts_pos, neg_exp)
            self.scaled_freq_neg *= (neg_ratio /
                                     np.sum(self.scaled_freq_neg, axis=0, keepdims=True))

    def minibatch(self, words, contexts):
        """Select a minibatch for all combinations of `words` and `contexts`

        Returns three numpy tensors with dtype `float32`:
        - Positive co-occurrence counts; shape `(len(words), len(contexts), num_timesteps)`.
        - Marginal positive counts; shape `(len(words), num_timesteps)`.
        - Maringal scaled negative frequencies; shape `(len(contexts), num_timesteps)`.
        """

        M, = words.shape
        N, = contexts.shape

        words_repeated = np.tile(words.reshape((-1, 1)), N).flatten()
        contexts_repeated = np.tile(contexts, M)
        offsets = np.empty((M * N,), dtype=self.indices.dtype)

        error_code = csr_sample_offsets(
            self.vocab_size, self.vocab_size, self.indptr, self.indices, M * N,
            words_repeated, contexts_repeated, offsets)
        assert error_code == 0

        pos = np.zeros((M * N, self.num_timesteps), dtype=self.data.dtype)
        for i, offset in enumerate(offsets):
            if offset != -1:
                start = self.indptr2[offset]
                stop = self.indptr2[offset + 1]
                pos[i, self.timesteps[start:stop]] = self.data[start:stop]

        pos = pos.reshape((M, N, self.num_timesteps))
        marginal_pos = self.counts_pos[words, :]
        scaled_freq_neg = self.scaled_freq_neg[contexts, :]

        return pos, marginal_pos, scaled_freq_neg
