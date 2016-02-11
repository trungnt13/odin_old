from __future__ import print_function, division

import numpy as np

MAGIC_SEED = 12082518
# ===========================================================================
# DAA
# ===========================================================================
class queue(object):

    """ FIFO, fast, NO thread-safe queue """

    def __init__(self):
        super(queue, self).__init__()
        self._data = []
        self._idx = 0

    def put(self, value):
        self._data.append(value)

    def append(self, value):
        self._data.append(value)

    def pop(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def get(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def empty(self):
        if self._idx == len(self._data):
            return True
        return False

    def clear(self):
        del self._data
        self._data = []
        self._idx = 0

    def __len__(self):
        return len(self._data) - self._idx

# ===========================================================================
# Utilities functions
# ===========================================================================
def segment_list(l, n_seg):
    '''
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> [[1], [2], [3], [4, 5]]
    '''
    # by floor, make sure and process has it own job
    size = int(np.ceil(len(l) / float(n_seg)))
    if size * n_seg - len(l) > size:
        size = int(np.floor(len(l) / float(n_seg)))
    # start segmenting
    segments = []
    for i in xrange(n_seg):
        start = i * size
        if i < n_seg - 1:
            end = start + size
        else:
            end = max(start + size, len(l))
        segments.append(l[start:end])
    return segments

def create_batch(n_samples, batch_size,
    start=None, end=None, prng=None, upsample=None, keep_size=False):
    '''
    No gaurantee that this methods will return the extract batch_size

    Parameters
    ----------
    n_samples : int
        size of original full dataset (not count start and end)
    prng : numpy.random.RandomState
        if prng != None, the upsampling process will be randomized
    upsample : int
        upsample > n_samples, batch will be sampled from original data to make
        the same total number of sample
        if [start] and [end] are specified, upsample will be rescaled according
        to original n_samples

    Example
    -------
    >>> from numpy.random import RandomState
    >>> create_batch(100, 17, start=0.0, end=1.0)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    >>> create_batch(100, 17, start=0.0, end=1.0, upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (0, 20), (20, 37)]
    >>> create_batch(100, 17, start=0.0, end=1.0, prng=RandomState(12082518), upsample=130)
    >>> [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (20, 40), (80, 90)]

    Notes
    -----
    If you want to generate similar batch everytime, set the same seed before
    call this methods
    For odd number of batch and block, a goal of Maximize number of n_block and
    n_batch are applied
    '''
    #####################################
    # 1. Validate arguments.
    if start is None or start >= n_samples or start < 0:
        start = 0
    if end is None or end > n_samples:
        end = n_samples
    if end < start: #swap
        tmp = start
        start = end
        end = tmp

    if start < 1.0:
        start = int(start * n_samples)
    if end <= 1.0:
        end = int(end * n_samples)
    orig_n_samples = n_samples
    n_samples = end - start

    if upsample is None:
        upsample = n_samples
    else: # rescale
        upsample = int(upsample * float(n_samples) / orig_n_samples)
    #####################################
    # 2. Init.
    jobs = []
    n_batch = float(n_samples / batch_size)
    if n_batch < 1 and keep_size:
        raise ValueError('Cannot keep size when number of data < batch size')
    i = -1
    for i in xrange(int(n_batch)):
        jobs.append((start + i * batch_size, start + (i + 1) * batch_size))
    if not n_batch.is_integer():
        if keep_size:
            jobs.append((end - batch_size, end))
        else:
            jobs.append((start + (i + 1) * batch_size, end))

    #####################################
    # 3. Upsample jobs.
    upsample_mode = True if upsample >= n_samples else False
    upsample_jobs = []
    n = n_samples if upsample_mode else 0
    i = 0
    while n < upsample:
        # pick a package
        # ===========================================================================
        # DAA
        # ===========================================================================
        if prng is None:
            added_job = jobs[i % len(jobs)]
            i += 1
        elif prng is not None:
            added_job = jobs[prng.randint(0, len(jobs))]
        tmp = added_job[1] - added_job[0]
        if not keep_size: # only remove redundant size if not keep_size
            if n + tmp > upsample:
                tmp = n + tmp - upsample
                added_job = (added_job[0], added_job[1] - tmp)
        n += added_job[1] - added_job[0]
        # done
        upsample_jobs.append(added_job)

    if upsample_mode:
        return jobs + upsample_jobs
    else:
        return upsample_jobs
