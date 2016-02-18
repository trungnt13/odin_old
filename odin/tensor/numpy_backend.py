# ===========================================================================
# Some functions of this module are adpated from: https://github.com/fchollet/keras
# Original work Copyright (c) 2014-2015 keras contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy as sp

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

# ===========================================================================
# Main
# ===========================================================================

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
   return result

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
