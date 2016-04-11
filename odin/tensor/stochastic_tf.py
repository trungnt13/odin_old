from __future__ import division

import tensorflow as tf
import numpy as np
from . import tf_backend as TB

from .. import config
from .numpy_backend import get_random_magic_seed

_FLOATX = config.floatX()
_EPSILON = config.epsilon()


# ===========================================================================
# RANDOMNESS
# ===========================================================================
class _RandomWrapper(object):

    def __init__(self, rng):
        super(_RandomWrapper, self).__init__()
        self._rng = np.random.RandomState(rng)
        self._state = np.random.RandomState(rng)

    def randint(self):
        return self._state.randint(10e6)

    def normal(self, shape, mean, std, dtype=_FLOATX):
        return tf.random_normal(shape=shape, mean=mean, stddev=std,
            dtype=dtype, seed=self._rng.randint(10e6))

    def uniform(self, shape, low, high, dtype=_FLOATX):
        return tf.random_uniform(shape=shape, minval=low, maxval=high,
                             dtype=dtype, seed=self._rng.randint(10e6))

    def binomial(self, shape, p, dtype=_FLOATX):
        return tf.cast(
            tf.less(
                tf.random_uniform(shape=shape, minval=0., maxval=1.,
                             dtype=_FLOATX, seed=self._rng.randint(10e6)),
                p),
            dtype)

    def shuffle(self, x):
        self._state.shuffle(x)


def rng(seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return _RandomWrapper(seed)


def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype, seed=seed)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype, seed=seed)


def random_binomial(shape, p, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return tf.cast(
        tf.less(
            tf.random_uniform(shape=shape, minval=0., maxval=1.,
                         dtype=dtype, seed=seed),
            p),
        dtype)


def dropout(x, level, rescale=True, noise_shape=None,
    seed=None, rng=None):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Parameters
    ----------
    x: A tensor.
    level: float(0.-1.)
        probability dropout values in given tensor
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: int
        A Python integer. Used to create random seeds. See
    rng: `tensor.rng`
        random generator from tensor class
    """
    retain_prob = 1. - level
    if isinstance(rng, _RandomWrapper):
        seed = rng._rng.randint(10e6)
    elif seed is None:
        seed = get_random_magic_seed()

    if noise_shape is not None:
        # from tensorflow.python.ops import array_ops
        # shape_x = array_ops.shape(x)
        noise_shape = tuple([TB.shape(x)[i].value if j is None or j < 0 else j
                            for i, j in enumerate(noise_shape)])
    # the dummy 1. works around a TF bug
    # (float32_ref vs. float32 incomptability)
    x = tf.nn.dropout(x * 1., retain_prob, noise_shape=noise_shape, seed=seed)
    if not rescale:
        x = x * retain_prob
    return x


# ===========================================================================
# Variational
# ===========================================================================
def kl_gaussian(mean_, logsigma,
                prior_mean=0., prior_logsigma=0.,
                regularizer_scale=1.):
    ''' KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer
    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
        log on logsigma)
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Note
    ----
    origin implementation from seya:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    '''
    gauss_klds = 0.5 * (prior_logsigma - logsigma +
            ((tf.exp(logsigma) + pow((mean_ - prior_mean), 2.0)) / tf.exp(prior_logsigma)) - 1.0)
    return TB.mean(gauss_klds)
