'''Module for defining user-controllable Tensorflow optimizers.'''

from .local_variable_optimizer import LocalAdamOptimizer

import tensorflow as tf


def add_cli_args(parser):
    '''Add command line arguments that control the optimization method.

    This function defines command line arguments that control an iterative
    optimization method. Currently, it allows choosing between standard SGD,
    Adam, and Adagrad, and to set the corresponding learning rates and momenta.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    The command line argument group that was added to the parser.
    '''

    group = parser.add_argument_group('Optimization Parameters')
    group.add_argument('--optimizer', metavar='OPT', default='adam',
                       choices=['sgd', 'adam', 'adagrad'],  help='''
        Set the Optimization method.''')
    group.add_argument('--lr0', metavar='FLOAT', type=float, default=0.1, help='''
        Set the initial prefactor for the (possibly adaptive) learning rate. See
        `--lr_exponent`.''')
    group.add_argument('--lr_offset', metavar='FLOAT', type=float, default=100.0, help='''
        Set the time scale on which the learning rate drops (unless `--lr_exponent` is set to
        zero). See `--lr_exponent`.''')
    group.add_argument('--lr_exponent', metavar='FLOAT', type=float, default=0.7, help='''
        Set the exponent by which the prefactor of the learning rate drops as a function of the
        training step. The learning rate in step $t$ is $\\rho_0 (a/(t+a))^b$, where the initial
        learning rate $\\rho_0$ is controlled by `--lr0`, the offset $a$ is controlled by
        `--lr_offset`, and the exponent $b$ is controlled by `--lr_exponent`. In order to satisfy
        the requirements by Robbins and Monro (1951), we recommended the following values; for
        `--optimizer sgd` and `--optimizer adam`:  $0.5 < b <= 1$; for `--optimizer adagrad`:
        $0 < b <= 0.5$.''')
    group.add_argument('--adam_beta1', metavar='FLOAT', type=float, default=0.9, help='''
        Only used for `--optimizer adam`. Set the decay rate of the first moment estimator in the
        Adam optimizer.''')
    group.add_argument('--adam_beta2', metavar='FLOAT', type=float, default=0.99, help='''
        Only used for `--optimizer adam`. Set the decay rate of the second moment estimator in the
        Adam optimizer.''')
    group.add_argument('--adam_epsilon', metavar='FLOAT', type=float, default=1e-8, help='''
        Only used for `--optimizer adam`. Set a regularizer to prevent division by zero in the
        Adam optimizer in edge cases.''')
    group.add_argument('--adagrad_init', metavar='FLOAT', type=float, default=0.1, help='''
        Only used for `--optimizer adagrad`. Set initial accumulator of the Adagrad optimizer.''')
    return group


def define_optimizer(loss, vars, args, minibatch=False):
    '''Create a tensorflow optimizer according to the provided command line arguments.

    Arguments:
    loss -- A Tensorflow scalar. The objective function.
    vars -- A list of the Tensorflow variables over which to optimizer. In the
        simplest case, use `tf.trainable_variables()` here.
    args -- A python namespace containing the parsed command line arguments.
        Must contain the arguments defined by the function `add_cli_args()` in
        the same module.
    '''

    # with tf.device('/CPU:0'):
    concluded_training_steps = tf.Variable(0, dtype=tf.int32, name='step')
    with tf.variable_scope('learning_rate'):
        step_float = tf.cast(concluded_training_steps, dtype=tf.float32)
        lr = args.lr0 * (args.lr_offset /
                         (step_float + args.lr_offset)) ** args.lr_exponent
    # with tf.device('/CPU:0'):
    tf.compat.v1.summary.scalar("learning_rate", lr)

    if args.optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(lr)
    elif args.optimizer == 'adam':
        if minibatch:
            opt = LocalAdamOptimizer(
                lr, args.adam_beta1, args.adam_beta2, args.adam_epsilon)
        else:
            opt = tf.compat.v1.train.AdamOptimizer(
                lr, args.adam_beta1, args.adam_beta2, args.adam_epsilon)
    elif args.optimizer == 'adagrad':
        opt = tf.compat.v1.train.AdagradOptimizer(lr, args.adagrad_init)
    else:
        raise 'Invalid optimizer `%s`.' % args.optimizer

    grads_and_vars = opt.compute_gradients(loss, var_list=vars)

    grads_and_vars = [(tf.clip_by_value(g, -0.1, 0.1), v)
                      for (g, v) in grads_and_vars]

    opt_step = opt.apply_gradients(
        grads_and_vars, global_step=concluded_training_steps, name='update_step')

    if args.debug and isinstance(opt, LocalAdamOptimizer):
        opt.local_updates_summary_ops()

    return opt_step


# TODO: Do we still need this?
def define_optimizer_apply_grad(grads, vars, args):
    '''Create a tensorflow optimizer according to the provided command line arguments.

    Arguments:
    grads -- A list of tensors that are gradients of vars.
    vars -- A list of the Tensorflow variables over which to optimizer.
    args -- A python namespace containing the parsed command line arguments.
        Must contain the arguments defined by the function `add_cli_args()` in
        the same module.
    '''

    concluded_training_steps = tf.Variable(0, dtype=tf.int32, name='step')
    with tf.variable_scope('learning_rate'):
        step_float = tf.cast(concluded_training_steps, dtype=tf.float32)
        lr = args.lr0 * (args.lr_offset /
                         (step_float + args.lr_offset)) ** args.lr_exponent

    if args.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif args.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(
            lr, args.adam_beta1, args.adam_beta2, args.adam_epsilon)
    elif args.optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer(lr, args.adagrad_init)
    else:
        raise 'Invalid optimizer `%s`.' % args.optimizer

    grads_and_vars = [(grad, var) for (grad, var) in zip(grads, vars)]

    return opt.apply_gradients(grads_and_vars,
                               global_step=concluded_training_steps, name='update_step')
