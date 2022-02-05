"""TODO: document"""

import numpy as np
import tensorflow as tf


class GoldstoneGdOptimizer:
    """TODO: document"""

    def __init__(self,
                 variable,
                 coupling_strengths,
                 name="GoldstoneGdOptimizer"):
        """TODO: document

        Args:
            variable: Float tensor of shape (vocab_size N, embedding_dim K, time_steps T).
        """
        _N, K, T = variable.shape.as_list()
        assert coupling_strengths.shape == (T - 1,)

        # Normalize average coupling strength to 1 (makes algorithm more numerically
        # stable without otherwise influencing the behavior)
        coupling_strengths *= (T - 1) / np.sum(coupling_strengths)

        g_raw = tf.Variable(
            tf.zeros((T, K, K), dtype=variable.dtype), trainable=False, name="gauge_fields")
        self._gauge_fields = g_raw - tf.transpose(g_raw, (0, 2, 1))

        learning_rate = tf.Variable(
            tf.zeros((), dtype=variable.dtype), name="learning_rate")

        with tf.device('/CPU:0'):
            # Add small positive number `1e-4 * K` in denominator to avoid division by zero.
            # The prefactor of `0.25` seems to be important here (optimization diverges without
            # it). This is different from the paper. Maybe the original code implicitly scaled
            # the gauge fields by a factor of 0.5, resulting in an effective factor of
            # 0.25 for the learning rate.
            new_learning_rate = (0.25 * K) / (1e-4 * K + tf.reduce_sum(variable**2))
        set_learning_rate = tf.compat.v1.assign(learning_rate, new_learning_rate)

        m_t = tf.Variable(
            tf.zeros((T-1, K, K), dtype=variable.dtype), name="m")

        with tf.device('/CPU:0'):
            # Calculate M matrix on the CPU so that we don't have to copy the entire `variable`
            # to the GPU.
            variable_t = tf.transpose(variable, (2, 1, 0))  # shape (T, K, N)
            m_t_op = tf.matmul(variable_t[1:], variable_t[:-1],
                               transpose_b=True)  # shape (T-1, K, K)

        set_m_t = tf.compat.v1.assign(m_t, m_t_op)

        clip_g = tf.compat.v1.assign(
            g_raw, tf.clip_by_value(g_raw, -0.5 / K, 0.5 / K))
        self._start_op = tf.group(
            set_m_t, set_learning_rate, clip_g,  name="start_goldstone_gd")

        g_bgn = self.gauge_fields[:-1]
        g_end = self.gauge_fields[1:]
        g_diff = coupling_strengths.reshape((T - 1, 1, 1)) * (g_end - g_bgn)

        loss_factor1 = (g_diff + 0.5 * (
                        tf.matmul(g_end, g_diff, transpose_b=True) -
                        tf.matmul(g_diff, g_bgn, transpose_b=True)))
        loss = tf.reduce_sum(loss_factor1 * m_t)

        inv_laplacian = self._laplacian_pseudo_inv(coupling_strengths)

        grad = tf.gradients(loss, g_raw)  # shape (T, K, K)
        grad_reshaped = tf.reshape(grad, (T, K**2))
        grad_step_reshaped = tf.matmul(
            (-learning_rate) * inv_laplacian, grad_reshaped)  # shape (T, K**2)
        grad_step = tf.reshape(grad_step_reshaped, (T, K, K))
        self._step_op = tf.compat.v1.assign_add(g_raw, grad_step)

        scaled_clipped_gauge_fields = 0.1 * tf.clip_by_value(
            self.gauge_fields, -1.0 / K, 1.0 / K)
        with tf.device('/CPU:0'):
            variable_t2 = tf.transpose(variable, (2, 0, 1))  # shape (T, N, K)
            apply_step_t2 = tf.matmul(
                variable_t2, scaled_clipped_gauge_fields, transpose_b=True)  # shape (T, N, K)

            self._apply_step = tf.transpose(
                apply_step_t2, (1, 2, 0))  # shape (N, K, T)

        if isinstance(variable, tf.Variable):
            self._apply_op = tf.compat.v1.assign_add(variable, self.apply_step)

        with tf.device('/CPU:0'):
            tf.compat.v1.summary.histogram("gauge_fields", self.gauge_fields)
            tf.compat.v1.summary.scalar("gauge_learning_rate", learning_rate)

    def _laplacian_pseudo_inv(self, coupling_strengths):
        orig_dtype = coupling_strengths.dtype
        coupling_strengths = coupling_strengths.astype(np.float64)
        laplacian = (np.diag(-coupling_strengths, 1) +
                     np.diag(-coupling_strengths, -1))
        laplacian -= np.diag(np.sum(laplacian, axis=1))

        eigvals, eigvecs = np.linalg.eigh(laplacian)
        assert np.allclose(eigvals[0], 0)
        assert np.all(eigvals[1:] > 0)
        assert np.allclose((eigvecs * eigvals).dot(eigvecs.T), laplacian)

        inv_eigvals = np.zeros_like(eigvals)
        # Add a regularizer: make sure that inv_eigvals are not larger than T
        T = len(eigvals)
        inv_eigvals[1:] = T / (1.0 + T * eigvals[1:])

        return (eigvecs * inv_eigvals).dot(eigvecs.T).astype(orig_dtype)

    def optimize(self, session, num_steps):
        session.run(self.start)
        for _ in range(num_steps):
            session.run(self.run_step)
        session.run(self.apply)

    @property
    def gauge_fields(self):
        return self._gauge_fields

    @property
    def start(self):
        return self._start_op

    @property
    def run_step(self):
        return self._step_op

    @property
    def apply_step(self):
        return self._apply_step

    @property
    def apply(self):
        return self._apply_op
