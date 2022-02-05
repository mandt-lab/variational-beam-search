'''Module for boilerplate code around the training loop.

Defines an abstract base class `Model` for Bayesian word embedding models, a
function `train()` that runs the training loop, and a function `add_cli_args()`
that adds command line arguments to control the training loop (e.g., the number
of training steps and the log frequency).
'''

import pickle
from time import time
import abc
import traceback
import datetime
import socket
import subprocess
import os
import sys
import pprint
import numpy as np

import tensorflow as tf
from tensorflow.python.client import timeline


def add_cli_args(parser):
    '''Add generic command line arguments.

    This function defines command line arguments that are required for all
    models in this project.

    Arguments:
    parser -- An `argparse.ArgumentParser`.

    Returns:
    A tuple of two command line argument groups that were added to the parser.
    '''

    positional_args = parser.add_argument_group(
        'Required positional arguments')
    positional_args.add_argument('input', metavar='IN_PATH', help='''
        Path to a binary file or directory containing the preprocessed data sets.''')
    positional_args.add_argument('output', metavar='OUT_PATH', help='''
        Path to the output directory. Must not already exist.''')

    train_args = parser.add_argument_group(
        'Parameters of the training environment')

    train_args.add_argument('-f', '--force', action='store_true', help='''
        Allow writing into existing output directory, possibly overwriting existing files.''')
    train_args.add_argument('-E', '--epochs', metavar='N', type=int, default=10000, help='''
        Set the number of training epochs.''')
    train_args.add_argument('--rng_seed', metavar='N', type=int, help='''
        Set the seed of the pseudo random number generator. If not provided, a
        seed will automatically be generated from a system random source. In order to
        make experiments reproducible, the seed is always written to the output file,
        along with the git commit hash and all command line arguments.''')
    train_args.add_argument('--steps_per_summary', metavar='N', type=int, default=1000, help='''
        Set the number of training steps to run between generating a Tensorboard summary.''')
    train_args.add_argument('--initial_summaries', metavar='N', type=int, default=100, help='''
        Set the number of initial training steps for which a Tensorboard summary will be generated
        after every step.''')
    train_args.add_argument('--steps_per_checkpoint', metavar='N', type=int, default=1000, help='''
        Set the number of training steps to run between saving checkpoints. A final saver
        will always be saved after the last regular training step. Set to zero if you only want to
        save the final saver.''')
    train_args.add_argument('--validation_input_dir', metavar='DIRECTORY_PATH', help='''
        Path to a binary file containing the preprocessed data set.''')
    train_args.add_argument('--debug', action='store_true', help='''
        Generate several different kinds of debugging output that involve expensive operations or
        generate a lot of files and that should therefore not be used in a production environment.
        In detail, --debug turns on the following output: (i) internal states of the optimizer that
        involves the entire batch (i.e., not just the current minibatch); (ii) profiling
        information (JSON files that can be loaded in Google Chrome's profiler to analyze the time
        spent on each operation); and (iii) device placements of all operations (written to
        stdout).''')
    return positional_args, train_args


def train(Model, arg_parser):
    '''Create an instance of model with the command line arguments and train it.

    Arguments:
    Model -- A class derived from `Model` in this module.
    arg_parser -- An `argparse.ArgumentParser`.
    '''

    args = arg_parser.parse_args()

    # Get random seed from system if the user did not specify a random seed.
    if args.rng_seed is None:
        args.rng_seed = int.from_bytes(os.urandom(4), byteorder='little')
    rng = np.random.RandomState(seed=args.rng_seed)
    tf.compat.v1.set_random_seed(rng.randint(2**31))

    # Create the output directory.
    try:
        os.mkdir(args.output)
    except OSError:
        if not args.force:
            sys.stderr.write(
                'ERROR: Cannot create output directory `%s`.\n' % args.output)
            sys.stderr.write(
                'HINT: Does the directory already exist? To prevent accidental data loss this\n'
                '      script, by default, does not write to an existing output directory.\n'
                '      Specify a non-existing output directory or use the `--force`.')
            exit(1)
    else:
        print('Writing output into directory `%s`.' % args.output)

    try:
        with open(os.path.join(args.output, 'log'), 'w') as log_file:
            # We write log files in the form of python scripts. This way, log files are both human
            # readable and very easy to parse by different python scripts. We begin log files with
            # a shebang (`#!/usr/bin/python`) so that text editors turn on syntax highlighting.
            log_file.write('#!/usr/bin/python\n')
            log_file.write('\n')

            # Log information about the executing environment to make experiments reproducible.
            log_file.write('program = "%s"\n' % arg_parser.prog)
            log_file.write(
                'args = {\n %s\n}\n\n' % pprint.pformat(vars(args), indent=2)[1:-1])
            # log_file.write('git_revision = "%s"\n' % subprocess.check_output(
            #     ['git', 'rev-parse', 'HEAD']).decode('utf-8').strip())
            log_file.write('host_name = "%s"\n' % socket.gethostname())
            log_file.write('using_gpu = %s\n' % tf.test.is_gpu_available())
            log_file.write('start_time = "%s"\n' %
                           str(datetime.datetime.now()))
            log_file.write('\n')

            model = Model(args, rng, log_file)
            model.fit(args, rng, log_file)

            log_file.write('\n')
            log_file.write('end_time = "%s"\n' %
                           str(datetime.datetime.now()))
    except:
        with open(os.path.join(args.output, 'err'), 'w') as err_file:
            traceback.print_exc(file=err_file)
        exit(2)


class Model(abc.ABC):
    '''Abstract base class of word embedding models.

    Concrete subclasses of `Model` define the model architecture in their
    initializer. They have to provide the attribute `opt_step`, which must
    return a tensorflow op that performs a single optimization step.

    Typically, the initializer will also load the training data, so instances
    really represent a Model paired with its training data.
    '''

    def fit(self, args, rng, log_file):
        '''Fit the model to its data.

        Arguments:
        args -- A python namespace containing the parsed command line
            parameters.
        rng -- An `np.random.RandomState`.
        log_file -- A file handle for log messages. Log messages will be written
            in the form of python statements so that the log file can easily be
            parsed by other python scripts.
        '''
        training_steps = args.epochs * self.steps_per_epoch
        log_file.write('\n')
        log_file.write('first_training_step = 1\n')
        log_file.write('last_training_step = %d\n' % training_steps)
        log_file.write(
            'progress_columns = ["step", "avg_duration", "of_which_avg_post_step_duration"]\n')
        log_file.write('\n')
        log_file.write('progress = [\n')
        log_file.flush()

        # with tf.device('/CPU:0'):
        summary_ops = tf.summary.merge_all()

        opt_step = self.opt_step
        if args.debug:
            session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            run_options = None
            run_metadata = None

        session.run([tf.initializers.global_variables(), tf.initializers.local_variables()],
                    feed_dict=self.init_feed_dict())
        # with tf.device('/CPU:0'):
        summary_writer = tf.summary.FileWriter(args.output, session.graph)
        saver = tf.train.Saver(
            tf.trainable_variables() + self.courtesy_checkpoint_variables, max_to_keep=1)

        time_last_summary = time()
        step_last_summary = 0
        duration_post_step_action = 0

        for step in range(1, training_steps + 1):
            if step % args.steps_per_summary == 0 or step <= args.initial_summaries:
                _, summary = session.run(
                    [opt_step, summary_ops], feed_dict=self.generate_feed_dict(rng),
                    options=run_options, run_metadata=run_metadata)
                new_time = time()
                avg_duration = ((new_time - time_last_summary) /
                                (step - step_last_summary))
                avg_post_step_duration = (duration_post_step_action /
                                          (step - step_last_summary))
                time_last_summary = new_time
                duration_post_step_action = 0
                step_last_summary = step
                log_file.write(
                    '(%d, %g, %g),\n' % (step, avg_duration, avg_post_step_duration))
                log_file.flush()

                # with tf.device('/CPU:0'):
                summary_writer.add_summary(summary, global_step=step)
            else:
                session.run(
                    opt_step, feed_dict=self.generate_feed_dict(rng),
                    options=run_options, run_metadata=run_metadata)

            if args.debug:
                # Create the Timeline object, and write it to a json file.
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('%s/timeline_%04d.json' % (args.output, step), 'w') as f:
                    f.write(chrome_trace)
                run_metadata = tf.RunMetadata()

            start_post_step_action = time()
            self.post_step_action(session, step, rng, log_file)
            duration_post_step_action += time() - start_post_step_action

            if step % args.steps_per_checkpoint == 0:
                self.save_checkpoint(args, session,
                                     step, saver, rng, log_file)
        log_file.write(']\n')

        if step % args.steps_per_checkpoint != 0:
            self.save_checkpoint(args, session,
                                 step, saver, rng, log_file)

        self.post_training_action(session, step, rng, log_file)

    def save_checkpoint(self, args, session, step, saver, rng, log_file):
        '''Save the current state of the model to a tensorflow checkpoint.

        Also saves the state of the random number generator.

        Arguments:
        args -- A python namespace containing the parsed command line
            parameters. Among the parameters, args.output is the output 
            directory. Must exist. Any files with clashing names
            in the output directory will be overwritten.
        session -- A `tf.Session` that contains the state.
        step -- Integer number of concluded training steps.
        saver -- A `tf.train.Saver`.
        rng -- An `np.random.RandomState`.
        log_file -- File handle to the log file.
        '''

        start_time = time()
        log_file.write('# Saving checkpoint... ')
        log_file.flush()

        # with tf.device('/CPU:0'):
        saver.save(session, os.path.join(args.output, 'checkpoint'),
                   global_step=step)

        new_rng_path = os.path.join(args.output, 'rng-%d.pickle' % step)
        with open(new_rng_path, 'wb') as f:
            pickle.dump(rng.get_state(), f)
        if hasattr(self, '_last_rng_path'):
            os.remove(self._last_rng_path)
        self._last_rng_path = new_rng_path

        if args.validation_input_dir is not None:
            self.validation(args, step)

        log_file.write('done. (%.2g seconds)\n' % (time() - start_time))
        log_file.flush()

    def validation(self, args, step):
        '''Validation step. 

        NOTE: once you use validation data, you should override this method in
        the derived child class.

        Arguments:
        args -- A python namespace containing the parsed command line
            parameters. Among the parameters, args.validation_input_dir is the 
            validation data path, which is used. 
        step -- Training steps after which the validation occurs.

        Raise:
        NotImplementedError -- If this method is called but not implemented in
            the child class, NotImplementedError exception is raised.
        '''
        raise NotImplementedError('Validation data is used but validation step \
            is not implemented.')

    @property
    def steps_per_epoch(self):
        '''Number of training steps per epoch.

        Overwrite this method if you train using minibatch sampling. It should
        return an integer number that indicates how many training steps
        correspond to one epoch. This only influences how the command line
        argument `--epochs` (or `-E`) is interperted.

        The default implementation returns `1`, which is suitable for full batch
        training.
        '''
        return 1

    @property
    @abc.abstractmethod
    def opt_step(self):
        '''A Tensorflow op to perform a single training step.

        Overwrite this method to return an appropriate op, usually the return
        value of the `minimize()` method of a Tensorflow optimizer. This op
        should perform a single training step without writing out Tensorboard
        summaries. Summary are handled automatically at appropriate steps.'''
        pass

    @property
    def courtesy_checkpoint_variables(self):
        return []

    def generate_feed_dict(self, rng):
        '''Generate a dict structure as an input to the tensorflow graph.

        Overwrite this method to return a dict used by the argument `feed_dict`
        of `sess.run()` method. 

        The default returns None.
        '''
        return None

    def init_feed_dict(self):
        return None

    def post_step_action(self, session, step, rng, log_file):
        pass

    def post_training_action(self, session, step, rng, log_file):
        pass
        # log_file.write('post_training_action')
