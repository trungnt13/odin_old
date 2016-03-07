# ===========================================================================
# This module is created based on the code from 2 libraries: Lasagne and keras
# Original work Copyright (c) 2014-2015 keras contributors
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy as sp
from ..config import floatX

# ===========================================================================
# RandomStates
# ===========================================================================
_MAGIC_SEED = 12082518
_SEED_GENERATOR = np.random.RandomState(_MAGIC_SEED)


def set_magic_seed(seed):
    global _MAGIC_SEED, _SEED_GENERATOR
    _MAGIC_SEED = seed
    _SEED_GENERATOR = np.random.RandomState(_MAGIC_SEED)


def get_magic_seed():
    return _MAGIC_SEED


def get_random_magic_seed():
    return _SEED_GENERATOR.randint(10e6)


def get_random_generator():
    return _SEED_GENERATOR

# ===========================================================================
# Main
# ===========================================================================


def is_ndarray(x):
    return isinstance(x, np.ndarray)


def np_masked_output(X, X_mask):
    '''
    Example
    -------
        X: [[1,2,3,0,0],
            [4,5,0,0,0]]
        X_mask: [[1,2,3,0,0],
                 [4,5,0,0,0]]
        return: [[1,2,3],[4,5]]
    '''
    res = []
    for x, mask in zip(X, X_mask):
        x = x[np.nonzero(mask)]
        res.append(x.tolist())
    return res


def np_one_hot(y, n_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    if not n_classes:
        n_classes = np.max(y) + 1
    Y = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def np_split_chunks(a, maxlen, overlap):
    '''
    Example
    -------
    >>> print(split_chunks(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 5, 1))
    >>> [[1, 2, 3, 4, 5],
         [4, 5, 6, 7, 8]]
    '''
    chunks = []
    nchunks = int((max(a.shape) - maxlen) / (maxlen - overlap)) + 1
    for i in xrange(nchunks):
        start = i * (maxlen - overlap)
        chunks.append(a[start: start + maxlen])

    # ====== Some spare frames at the end ====== #
    wasted = max(a.shape) - start - maxlen
    if wasted >= (maxlen - overlap) / 2:
        chunks.append(a[-maxlen:])
    return chunks


def np_ordered_set(seq):
    seen = {}
    result = []
    for marker in seq:
        if marker in seen: continue
        seen[marker] = 1
        result.append(marker)
    return np.asarray(result)


def np_shrink_labels(labels, maxdist=1):
    '''
    Example
    -------
    >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 1))
    >>> [0, 1, 0, 1, 0, 4, 5, 4, 6, 0]
    >>> print(shrink_labels(np.array([0, 0, 1, 0, 1, 1, 0, 0, 4, 5, 4, 6, 6, 0, 0]), 2))
    >>> [0, 1, 0, 4, 6, 0]
    Notes
    -----
    Different from ordered_set, the resulted array still contain duplicate
    if they a far away each other.
    '''
    maxdist = max(1, maxdist)
    out = []
    l = len(labels)
    i = 0
    while i < l:
        out.append(labels[i])
        last_val = labels[i]
        dist = min(maxdist, l - i - 1)
        j = 1
        while (i + j < l and labels[i + j] == last_val) or (j < dist):
            j += 1
        i += j
    return out

# ===========================================================================
# Special random algorithm for weights initialization
# ===========================================================================


def np_normal(shape, mean=0., std=1.):
    return np.cast[floatX()](
        get_random_generator().normal(mean, std, size=shape))


def np_constant(shape, val=0.):
    return np.cast[floatX()](np.zeros(shape) + val)


def np_symmetric_uniform(shape, range=0.01, std=None, mean=0.0):
    if std is not None:
        a = mean - np.sqrt(3) * std
        b = mean + np.sqrt(3) * std
    else:
        try:
            a, b = range  # range is a tuple
        except TypeError:
            a, b = -range, range  # range is a number
    return np.cast[floatX()](
        get_random_generator().uniform(low=a, high=b, size=shape))


def np_glorot_uniform(shape, gain=1.0, c01b=False):
    orig_shape = shape
    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
    else:
        if len(shape) < 2:
            shape = (1,) + tuple(shape)
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    a = 0.0 - np.sqrt(3) * std
    b = 0.0 + np.sqrt(3) * std
    return np.cast[floatX()](
        get_random_generator().uniform(low=a, high=b, size=orig_shape))


def np_glorot_normal(shape, gain=1.0, c01b=False):
    orig_shape = shape
    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        n1, n2 = shape[0], shape[3]
        receptive_field_size = shape[1] * shape[2]
    else:
        if len(shape) < 2:
            shape = (1,) + tuple(shape)
        n1, n2 = shape[:2]
        receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    return np.cast[floatX()](
        get_random_generator().normal(0.0, std, size=orig_shape))


def np_he_normal(shape, gain=1.0, c01b=False):
    if gain == 'relu':
        gain = np.sqrt(2)

    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        fan_in = np.prod(shape[:3])
    else:
        if len(shape) <= 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])

    std = gain * np.sqrt(1.0 / fan_in)
    return np.cast[floatX()](
        get_random_generator().normal(0.0, std, size=shape))


def np_he_uniform(shape, gain=1.0, c01b=False):
    if gain == 'relu':
        gain = np.sqrt(2)

    if c01b:
        if len(shape) != 4:
            raise RuntimeError(
                "If c01b is True, only shapes of length 4 are accepted")
        fan_in = np.prod(shape[:3])
    else:
        if len(shape) <= 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])

    std = gain * np.sqrt(1.0 / fan_in)
    a = 0.0 - np.sqrt(3) * std
    b = 0.0 + np.sqrt(3) * std
    return np.cast[floatX()](
        get_random_generator().uniform(low=a, high=b, size=shape))


def np_orthogonal(shape, gain=1.0):
    if gain == 'relu':
        gain = np.sqrt(2)

    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = get_random_generator().normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.cast[floatX()](gain * q)
