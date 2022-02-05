"""Stochastic optimizers for models with local latent variables.

This module contains a variant of optimizers in the `tf.train` name space.
Currently, only a variant of `tf.train.AdamOptimizer` is implemented.  The
difference to the standard Adam optimizer is that the implementation in this
module can efficiently update local latent variables.  Here, we call a variable
"local" if it affects only a single data point (e.g., in a nonparametric
model).  This has nothing to do with TensorFlow's notion of local variables
(= variables that are not saved by default).

Many standard gradient optimizers in TensorFlow, such as
`tf.train.AdamOptimizer`, are inefficient for minibatch training of models with
local variables.  They always update *all* entries of the trainable variables,
even those that do not affect the loss function estimator for the current
minibatch.  This is bad for two reasons.  First, it is inefficient to
explicitly apply a zero valued gradient step to all out-of-minibatch local
variables (making the cost of the update step proportional to the size of the
data set, rather than the size of the minibatch).  Second, optimizers that
estimate the gradient noise (such as Adam) interpret the zero gradients as a
valid gradient signal.  The estimated gradient noise is therefore always at
least as large as the average gradient itself.  This leads to slow convergence
due to a small adaptive learning rate.

The classes in this module implement stochastic optimizers that update only the
entries of the trainable variables that are actually affected by the minibatch.
This is done by changing the way sparse updates are performed in the
`apply_gradients` method.  It affects updates of all variables `var` that enter
the loss function only via a tensor returned from `tf.gather(var, ...)`.
"""

import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import init_ops


class LocalAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Variant of Adam optimizer that treats local variables more efficiently.

    Drop-in replacement for `tf.train.AdamOptimizer`.  The two optimizers
    differ in the treatment of local variables, i.e., variables `var` that
    enter the loss only via a tensor returned from `tf.gather(var, ...)`.
    The standard Adam optimizer updates the first and second moment estimates
    of the entire variable `var`, and then updates the entire `var`.  By
    contrast, this optimizer updates only the "gathered" components of `var`
    and of its first and second moment estimates.  It keeps track of the
    individual training step of each component of a local variable in order to
    properly normalize the adaptive learning rates.

    Any variable that enters the loss function directly (without `tf.gather`)
    is interpreted as a global variable and treated the same way as in the
    standard Adam optimizer.

    See module docstring for further discussion of local variable optimizers.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                 local_learning_rate=None, local_beta1=None, local_beta2=None,
                 use_locking=False, name='LocalAdam'):
        """Initialize a local variable Adam optimizer.

        See also `tf.train.AdamOptimizer.__init__`.

        Args:
            learning_rate (float): Learning rate for global variables and,
                unless `local_learning_rate` is given, also for local
                variables.
            beta1 (float): Decay rate of first moment estimates for
                global variables.
            beta2 (float): Decay rate of second moment estimates for
                global variables.
            local_learning_rate (float or None): Learning rate for local
                variables.  Defaults to `learning_rate`.  Note that for many
                probabilistic models (for which this optimizer is primarily
                intended), a decaying learning rate is recommended.  This holds
                in particular for variational inference, because there is
                typically no need to inject additional noise for regularization
                since VI already injects noise (even for a given minibatch) in
                a more principled way.
            local_beta1 (float or None): Decay rate of first moment estimates for
                local variables. Defaults to `beta1`.
            local_beta2 (float or None): Decay rate of second moment estimates for
                local variables. Defaults to `beta1`.
            epsilon (float): Small value to prevent division by zero.
            use_locking (float): See initializer of `tf.train.AdamOptimizer`.
            name (float): Name space.
        """
        super(LocalAdamOptimizer, self).__init__(
            learning_rate, beta1, beta2, epsilon,
            use_locking, name)
        self._idxs_beta12_powers = {}
        self._local_update_steps = {}
        self._local_neg_lrs = {}
        self._loc_lr = (learning_rate if local_learning_rate is None
                        else local_learning_rate)
        self._loc_beta1 = (beta1 if local_beta1 is None else local_beta1)
        self._loc_beta2 = (beta2 if local_beta2 is None else local_beta2)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._minus_loc_lr_t = None
        self._loc_beta1_t = None
        self._loc_beta2_t = None

    def _prepare(self):
        tf.train.AdamOptimizer._prepare(self)
        self._minus_loc_lr_t = ops.convert_to_tensor(
            -self._loc_lr, name="minus_local_learning_rate")
        self._loc_beta1_t = ops.convert_to_tensor(
            self._loc_beta1, name="local_beta1")
        self._loc_beta2_t = ops.convert_to_tensor(
            self._loc_beta2, name="local_beta2")

    def _get_local_beta_accum(self, var, indices):
        dtype = var.dtype.base_dtype
        shape = tf.TensorShape([var.shape[0]] + [1] * (len(var.shape) - 1))

        with tf.device('/CPU:0'):
            init1 = init_ops.Constant(self._loc_beta1, dtype=dtype)
            beta1_power = self._get_or_make_slot_with_initializer(
                var, init1, shape, dtype, "beta1_power", "beta1_power")
            init2 = init_ops.Constant(self._loc_beta2, dtype=dtype)
            beta2_power = self._get_or_make_slot_with_initializer(
                var, init2, shape, dtype, "beta2_power", "beta2_power")

        self._idxs_beta12_powers[var] = (indices, beta1_power, beta2_power)
        return beta1_power, beta2_power

    def _apply_sparse_shared(self, grad, var, indices, scat_add, scat_upd):
        beta1_power, beta2_power = self._get_local_beta_accum(var, indices)
        b1_pow = tf.gather(beta1_power, indices, validate_indices=False)
        b2_pow = tf.gather(beta2_power, indices, validate_indices=False)
        minus_lr_t = math_ops.cast(self._minus_loc_lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._loc_beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._loc_beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        minus_lr = (minus_lr_t * math_ops.sqrt(1 - b2_pow) / (1 - b1_pow))
        self._local_neg_lrs[var] = minus_lr

        # Sparse version of: m_t = beta1 * m + (1 - beta1) * g_t
        with tf.device('/CPU:0'):
            m = self.get_slot(var, "m")
            m_gathered = tf.gather(m, indices, validate_indices=False)
        m_scaled = m_gathered * beta1_t
        m_scaled_g_values = grad * (1 - beta1_t)
        m_new = m_scaled + m_scaled_g_values
        with tf.device('/CPU:0'):
            m_t = scat_upd(m, indices, m_new)

        # Sparse version of: v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        with tf.device('/CPU:0'):
            v = self.get_slot(var, "v")
            v_gathered = tf.gather(v, indices, validate_indices=False)
        v_scaled = v_gathered * beta2_t
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_new = v_scaled + v_scaled_g_values
        with tf.device('/CPU:0'):
            v_t = scat_upd(v, indices, v_new)

        upd = minus_lr * m_new / (math_ops.sqrt(v_new) + epsilon_t)
        self._local_update_steps[var] = upd
        with tf.device('/CPU:0'):
            var_update = scat_add(var, indices, upd)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _scatter_add(self, x, i, v):
        with tf.device('/CPU:0'):
            return state_ops.scatter_add(x, i, v, use_locking=self._use_locking)

    def _scatter_update(self, x, i, v):
        with tf.device('/CPU:0'):
            return state_ops.scatter_update(x, i, v, use_locking=self._use_locking)

    def _resource_scatter_update(self, x, i, v):
        with tf.device('/CPU:0'):
            with ops.control_dependencies(
                    [resource_variable_ops.resource_scatter_update(
                     x.handle, i, v)]):
                return x.value()

    def _apply_sparse(self, grad, var):
        with tf.device('/CPU:0'):
            return self._apply_sparse_shared(
                grad.values, var, grad.indices, self._scatter_add,
                self._scatter_update)

    def _resource_apply_sparse(self, grad, var, indices):
        with tf.device('/CPU:0'):
            return self._apply_sparse_shared(
                grad, var, indices, self._resource_scatter_add,
                self._resource_scatter_update)

    def local_updates_summary_ops(self, var_list=None, mean_abs=True, hist=True):
        """Create and return summary ops for the update steps of local vars.

        WARNING: This op creates summaries over *all* variables optimized
        variables, not just over the current minibatch. Therefore, all operations
        are performed on the CPU. This can be very expensive and should only be
        used for debuggin purpose.

        Returns an op that creates summaries that can be written to a file with
        a `tf.summary.FileWriter`, and inspected in Tensorboard.  The summaries
        contain statistics of the update steps of local variables (including
        the scaling with the negative learning rate).

        Call `minimize` or `apply_gradients` before calling this method.

        Args:
            var_list (list of tensors or None): List of local variables for
                which to create summary notes.  All variables in this list have
                to have been either part of the `var_list` argument in a
                previous call to `minimize`, or part of the `grads_and_vars`
                argument in a previous call to `apply_gradients`.  They have to
                be local variables, i.e., enter the loss function only via the
                return value of `tf.gather`.  Defaults to all local variables
                whose gradients are applied.
            mean_abs (bool): Whether or not to create a scalar summary for the
                mean absolute value of the update steps for each variable in
                `var_list`.
            hist (bool): Whether or not to create a histogram summary for the
                update steps of each variable in `var_list`.  Here, the update
                steps are the values that are *added* to the variables in
                `var_list`, i.e., a positive value in the histogram typically
                corresponds to a negative gradient of the loss.

        Returns:
            A single op that creates all requested summaries.

        """
        with tf.device('/CPU:0'):
            if var_list is None:
                var_list = self._local_update_steps.keys()
            with ops.name_scope(self._name):
                summaries = []
                for var in var_list:
                    basename = re.search('([^/:]+)[^/]*$', var.name).group(1)
                    upd = self._local_update_steps[var]
                    log_lr = tf.log(1e-8 - self._local_neg_lrs[var])
                    m = self.get_slot(var, "m")
                    v = self.get_slot(var, "v")
                    log_abs_m = tf.log(1e-8 + tf.abs(m))
                    log_sqrt_v = 0.5 * tf.log(1e-8 + v)
                    for label, op, scalar_op in (
                            ('log_lr', log_lr, log_lr),
                            ('log_abs_moment1', log_abs_m, log_abs_m),
                            ('log_sqrt_moment2', log_sqrt_v, log_sqrt_v),
                            ('update_step', upd, tf.abs(upd))):
                        with ops.colocate_with(upd), ops.name_scope(basename):
                            if mean_abs:
                                scalar_mean = tf.reduce_mean(scalar_op)
                                summaries.append(tf.summary.scalar(
                                    "mean_%s" % label, scalar_mean))
                            if hist:
                                summaries.append(tf.summary.histogram(
                                    label, op))
                return tf.compat.v1.summary.merge(summaries, name="summaries")

    def _finish(self, update_ops, name_scope):
        # Update the global power accumulators.
        newops = [tf.train.AdamOptimizer._finish(self, update_ops, name_scope)]
        # Update the local power accumulators.
        with ops.control_dependencies(update_ops):
            for idxs, b1pow, b2pow in self._idxs_beta12_powers.values():
                with ops.colocate_with(b1pow):
                    dtype = b1pow.dtype.base_dtype
                    b1 = math_ops.cast(self._loc_beta1_t, dtype)
                    b2 = math_ops.cast(self._loc_beta2_t, dtype)
                    with tf.device('/CPU:0'):
                        b1pow_old = tf.gather(b1pow, idxs, validate_indices=False)
                        b2pow_old = tf.gather(b2pow, idxs, validate_indices=False)
                        newops.append(state_ops.scatter_update(
                            b1pow, idxs, b1 * b1pow_old,
                            use_locking=self._use_locking))
                        newops.append(state_ops.scatter_update(
                            b2pow, idxs, b2 * b2pow_old,
                            use_locking=self._use_locking))
        with ops.name_scope(name_scope):
            return control_flow_ops.group(*newops, name="update")
