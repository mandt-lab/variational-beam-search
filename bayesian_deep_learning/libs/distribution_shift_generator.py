import struct
import sys
import pickle
import abc
import gzip
from copy import deepcopy
import numpy as np

import cv2
from albumentations import ShiftScaleRotate, ElasticTransform, HorizontalFlip
from albumentations import VerticalFlip, Compose
# if we want to apply deterministic elastic transformations, use the following
# API and set the numpy random number generator
from albumentations.augmentations.functional import elastic_transform
import tensorflow as tf
import random


class LongShiftScaleRotateTransformedGenerator(abc.ABC):
    """Abstract class with sequential transformation declared.
    After initiated with specific dataset, it can generates batches of examples
    with declared transformations.
    """
    @property
    @abc.abstractmethod
    def X_train(self):
        pass

    @property
    @abc.abstractmethod
    def Y_train(self):
        pass

    @property
    @abc.abstractmethod
    def X_test(self):
        pass

    @property
    @abc.abstractmethod
    def Y_test(self):
        pass

    @property
    @abc.abstractmethod
    def out_dim(self):
        # Total number of unique classes
        pass

    def __init__(self, rng=None, changerate=1, max_iter=10, task_size=2048,
                 validation=False):
        self.max_iter = max_iter
        self.cur_iter = 0
        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng
        self.validation = validation

        change_pos = list(range(0, max_iter+1, changerate))
        change_pos = change_pos[1:]

        # at these indices the dataset will be permuted
        self.switch_points = [j for j in change_pos if j <= self.max_iter]
        self.tasks_to_test = [0] + self.switch_points
        self.examples_per_iter = task_size # 1 # 2048 # 10000 # 2048 # 1024

        # for demonstrations
        scale_limits, rotate_limits, shift_limits = [], [], []

        # First task is without transformations
        self.transformers = [None]
        for i, _ in enumerate(self.switch_points):
            scale_limit = self.rng.normal(0, 0.3)
            # rotate_limit = self.rng.uniform(-180, 180) # -180~180
            rotate_limit = self.rng.normal(0, 10) # -30~30
            shift_limit = self.rng.choice([-1, 1]) * self.rng.beta(1, 10)
            scale_limits.append(scale_limit)
            rotate_limits.append(rotate_limit)
            shift_limits.append(shift_limit)
            ssr = ShiftScaleRotate(
                shift_limit=(shift_limit, shift_limit), 
                scale_limit=(scale_limit, scale_limit), 
                rotate_limit=(rotate_limit, rotate_limit), 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0.0, 
                p=1.0,
            )
            pipe = ssr
            self.transformers.append(pipe)
        # First task is (unpermuted) MNIST, subsequent tasks are random
        # permutations of pixels
        self.perm_indices = [list(range(self.X_train.shape[1]))]
        for i, _ in enumerate(self.switch_points):
            perm_inds = list(range(self.X_train.shape[1]))
            self.rng.shuffle(perm_inds)
            self.perm_indices.append(perm_inds)
        # make sure they are different permutations
        assert(len(set(tuple(perm_inds) for perm_inds in self.perm_indices)) 
            == len(self.perm_indices))

        self.idx_map = {}
        self.batch_indices = []
        last_switch_point = 0
        for i, switch_point in enumerate((self.switch_points 
                                            + [self.max_iter])):
            batch_inds = list(range(self.X_train.shape[0]))
            self.rng.shuffle(batch_inds)
            batch_inds = np.tile(batch_inds, 2) # for repetition
            for j in range(last_switch_point, switch_point):
                self.idx_map[j] = i
                # deal with repetition
                lbd = (j-last_switch_point)*self.examples_per_iter
                ubd = (j-last_switch_point+1)*self.examples_per_iter
                redundant_len = ((lbd//self.X_train.shape[0]) 
                    * self.X_train.shape[0])
                # update lower and upper bound
                lbd = lbd - redundant_len
                ubd = ubd - redundant_len
                self.batch_indices.append(batch_inds[lbd:ubd])
            last_switch_point = switch_point

        # np.save('./transform_params.npy', 
        #     np.asarray([scale_limits, rotate_limits, shift_limits]))

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def transform(self, transformer, images):
        '''
        Parameters:
            transformer - transformation taken from `albumentations'
            images - numpy array of shape (?, height*width) and assume 
                height==width for MNIST
        '''
        if transformer is None:
            # do not transform
            return images
        else:
            res_images = []
            for image in images:
                image = transformer(image=image)['image']
                res_images.append(image)
            return np.asarray(res_images)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            transformer = self.transformers[self.idx_map[self.cur_iter]]
            batch_inds = self.batch_indices[self.cur_iter]

            # Retrieve train data
            next_x_train = self.transform(
                transformer, 
                deepcopy(self.X_train[batch_inds, ...])
            )

            next_y_train = self.Y_train[batch_inds]

            # Retrieve test data
            next_x_test = self.transform(
                transformer,
                deepcopy(self.X_test)
            )

            next_y_test = self.Y_test
            if self.validation:
                # use first 5000 images as validation set
                next_x_test = next_x_test[:5000]
                next_y_test = next_y_test[:5000]
                print("Use first 5000 test images as validation set.")
            else:
                next_x_test = next_x_test[5000:]
                next_y_test = next_y_test[5000:]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def reset(self):
        self.cur_iter = 0


class LongElasticTransformedGenerator(abc.ABC):
    """Abstract class with sequential transformation declared.
    After initiated with specific dataset, it can generates batches of examples
    with declared transformations.
    """
    @property
    @abc.abstractmethod
    def X_train(self):
        pass

    @property
    @abc.abstractmethod
    def Y_train(self):
        pass

    @property
    @abc.abstractmethod
    def X_test(self):
        pass

    @property
    @abc.abstractmethod
    def Y_test(self):
        pass

    @property
    @abc.abstractmethod
    def out_dim(self):
        # Total number of unique classes
        pass

    def __init__(self, rng=None, changerate=1, max_iter=10, task_size=2048):
        self.max_iter = max_iter
        self.cur_iter = 0
        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng

        change_pos = list(range(0, max_iter+1, changerate))
        change_pos = change_pos[1:]

        # at these indices the dataset will be permuted
        self.switch_points = [j for j in change_pos if j <= self.max_iter]
        self.tasks_to_test = [0] + self.switch_points
        self.examples_per_iter = task_size # 1 # 2048 # 10000 # 2048 # 1024

        # First task is without transformations
        self.transformer_rng_seeds = [None]
        for i, switch_id in enumerate(self.switch_points):
            # use the step as seed
            self.transformer_rng_seeds.append(switch_id)

        np.save('./transform_seeds.npy', self.transformer_rng_seeds)

        self.idx_map = {}
        self.batch_indices = []
        last_switch_point = 0
        for i, switch_point in enumerate((self.switch_points 
                                            + [self.max_iter])):
            batch_inds = list(range(self.X_train.shape[0]))
            self.rng.shuffle(batch_inds)
            batch_inds = np.tile(batch_inds, 2) # for repetition
            for j in range(last_switch_point, switch_point):
                self.idx_map[j] = i
                # deal with repetition
                lbd = (j-last_switch_point)*self.examples_per_iter
                ubd = (j-last_switch_point+1)*self.examples_per_iter
                redundant_len = ((lbd//self.X_train.shape[0]) 
                    * self.X_train.shape[0])
                # update lower and upper bound
                lbd = lbd - redundant_len
                ubd = ubd - redundant_len
                self.batch_indices.append(batch_inds[lbd:ubd])
            last_switch_point = switch_point

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def transform(self, rng_seed, images):
        '''
        Parameters:
            rng_seed - seed for numpy.random.RandomState(). It ensures all 
                images use the same deterministic transformation.
            images - numpy array of shape (?, height*width) and assume 
                height==width for MNIST
        '''
        if rng_seed is None:
            # do not transform
            return images
        else:
            res_images = []
            for image in images:
                # reset to enable deterministic behaviour
                self.rng.seed(rng_seed)
                image = elastic_transform(
                    image,
                    sigma=4,
                    alpha=34,
                    alpha_affine=1,
                    random_state=self.rng
                )
                res_images.append(image)
            return np.asarray(res_images)

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            rng_seed = self.transformer_rng_seeds[self.idx_map[self.cur_iter]]
            batch_inds = self.batch_indices[self.cur_iter]

            # Retrieve train data
            next_x_train = self.transform(
                rng_seed, 
                deepcopy(self.X_train[batch_inds, ...])
            )

            next_y_train = self.Y_train[batch_inds]

            # Retrieve test data
            next_x_test = self.transform(
                rng_seed,
                deepcopy(self.X_test)
            )

            next_y_test = self.Y_test

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def reset(self):
        self.cur_iter = 0

class LongTransformedCifar10Generator(LongShiftScaleRotateTransformedGenerator):
    # load data
    (x_train, y_train), (x_test, y_test) = \
        tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    y_train = np.squeeze(y_train)
    x_test = x_test.astype('float32')
    y_test = np.squeeze(y_test)
    x_train /= 255
    x_test /= 255

    # Define train and test data
    X_train = x_train
    Y_train = y_train
    X_test = x_test
    Y_test = y_test

    # Total number of unique classes
    out_dim = 10

    def __init__(self, rng=None, changerate=1, max_iter=10, task_size=2048,
                 validation=False):
        super().__init__(rng, changerate, max_iter, task_size, validation)

class LongTransformedSvhnGenerator(LongShiftScaleRotateTransformedGenerator):
    # load data
    (x_train, y_train), (x_test, y_test) = np.load("./dataset/svhn.npy", 
                                                   allow_pickle=True)
    x_train = x_train.astype('float32')
    y_train = np.squeeze(y_train)
    x_test = x_test.astype('float32')
    y_test = np.squeeze(y_test)
    x_train /= 255
    x_test /= 255

    # Define train and test data
    X_train = x_train
    Y_train = y_train
    X_test = x_test
    Y_test = y_test

    # Total number of unique classes
    out_dim = 10

    def __init__(self, rng=None, changerate=1, max_iter=10, task_size=2048,
                 validation=False):
        super().__init__(rng, changerate, max_iter, task_size, validation)

