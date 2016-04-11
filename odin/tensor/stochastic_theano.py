from __future__ import division

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from .numpy_backend import get_random_magic_seed

from .. import config

_FLOATX = config.floatX()
_EPSILON = config.epsilon()
PI = np.pi
C = -0.5 * np.log(2 * PI)


# ===========================================================================
# RANDOMNESS
# ===========================================================================
class _RandomWrapper(object):

    def __init__(self, rng, state):
        super(_RandomWrapper, self).__init__()
        self._rng = rng
        self._state = state

    def randint(self):
        return self._state.randint(10e6)

    def normal(self, shape, mean, std, dtype=_FLOATX):
        return self._rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

    def uniform(self, shape, low, high, dtype=_FLOATX):
        return self._rng.uniform(size=shape, low=low, high=high, dtype=dtype)

    def binomial(self, shape, p, dtype=_FLOATX):
        return self._rng.binomial(size=shape, n=1, p=p, dtype=dtype)

    def shuffle(self, x):
        self._state.shuffle(x)


def rng(seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    return _RandomWrapper(RandomStreams(seed=seed),
                          np.random.RandomState(seed=seed))


def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = get_random_magic_seed()
    rng = RandomStreams(seed=seed)
    return rng.binomial(size=shape, n=1, p=p, dtype=dtype)

'''
more TODO:

tensordot -> soon to be introduced in TF
batched_tensordot -> reimplement
'''


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
    # ====== Validate arguments ====== #
    if seed is None:
        seed = get_random_magic_seed()
    if rng is None:
        rng = _RandomWrapper(RandomStreams(seed=seed),
                             np.random.RandomState(seed=seed))
    elif isinstance(rng, RandomStreams):
        rng = _RandomWrapper(rng, np.random.RandomState(seed=seed))
    # ====== Dropout ====== #
    retain_prob = 1. - level
    if noise_shape is None:
        x = x * rng.binomial(shape=x.shape, p=retain_prob, dtype=x.dtype)
    else:
        # validate remove all None or -1 dimension
        noise_shape = tuple([x.shape[i] if j is None or j < 0 else j
                       for i, j in enumerate(noise_shape)])
        # auto select broadcast shape
        broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
        if len(broadcast) > 0:
            x = x * T.addbroadcast(
                rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype),
                *broadcast)
        else:
            x = x * rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype)
    if rescale:
        x /= retain_prob
    return x


# ===========================================================================
# Variational OPERATIONS
# ===========================================================================
def log_prob_bernoulli(p_true, p_approx, mask=None):
    """
    Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, p_approx.shape[1]))
    log_prob_1 = p_true * T.log(p_approx)
    log_prob_0 = (1.0 - p_true) * T.log(1.0 - p_approx)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = T.sum((log_prob_01 * mask), axis=1, keepdims=True)
    return row_log_probs

#logpxz = -0.5*np.log(2 * np.pi) - log_sigma_decoder - (0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)


def log_prob_gaussian(mu_true, mu_approx, les_sigmas=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and standard deviations given by les_sigmas.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - T.log(T.abs_(les_sigmas)) - \
    ((mu_true - mu_approx)**2.0 / (2.0 * les_sigmas**2.0))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs


def log_prob_gaussian2(mu_true, mu_approx, log_vars=1.0, mask=None):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and log variances given by les_logvars.
    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    if mask is None:
        mask = T.ones((1, mu_approx.shape[1]))
    ind_log_probs = C - (0.5 * log_vars) - \
    ((mu_true - mu_approx)**2.0 / (2.0 * T.exp(log_vars)))
    row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
    return row_log_probs


def kl_gaussian(mu_left, logvar_left,
                mu_right=0., logvar_right=0.):
    ''' KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer
    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
        log on logsigma)

    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    '''
    gauss_klds = 0.5 * (logvar_right - logvar_left +
            (T.exp(logvar_left) / T.exp(logvar_right)) +
            ((mu_left - mu_right)**2.0 / T.exp(logvar_right)) - 1.0)
    return gauss_klds
