# ===========================================================================
# Some functions of this module are adpated from: https://github.com/fchollet/keras
# Original work Copyright (c) 2014-2015 keras contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import division, absolute_import, print_function

import numpy as np
import scipy as sp

def np_pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''
    Pad each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer than maxlen is truncated
    to maxlen. Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    Parameters:
    -----------
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
    -------
    x: numpy array with dimensions (number_of_sequences, maxlen)

    Example:
    -------
        > pad_sequences([[1,2,3],
                         [1,2],
                         [1,2,3,4]], maxlen=3, padding='post', truncating='pre')
        > [[1,2,3],
           [1,2,0],
           [2,3,4]]
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

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
