from __future__ import print_function, division, absolute_import

import os
import numpy as np
from numpy.random import RandomState

from .base import OdinObject
from .utils import create_batch, queue
from .ie import get_file
from .tensor import get_random_magic_seed
from . import logger

from six.moves import zip_longest
from collections import defaultdict
import h5py

__all__ = [
    'batch',
    'dataset'
]

# ===========================================================================
# Helper function
# ===========================================================================


def _auto_batch_size(shape):
    # This method calculate based on reference to imagenet size
    batch = 256
    ratio = np.prod(shape[1:]) / (224 * 224 * 3)
    batch /= ratio
    return 2**int(np.log2(batch))


def _get_chunk_size(shape, size):
    if size == 'auto':
        return True
    return (2**int(np.ceil(np.log2(size))),) + shape[1:]


def _hdf5_get_all_dataset(hdf, fileter_func=None, path='/'):
    res = []
    # init queue
    q = queue()
    for i in hdf[path].keys():
        q.put(i)
    # get list of all file
    while not q.empty():
        p = q.pop()
        if 'Dataset' in str(type(hdf[p])):
            if fileter_func is not None and not fileter_func(p):
                continue
            res.append(p)
        elif 'Group' in str(type(hdf[p])):
            for i in hdf[p].keys():
                q.put(p + '/' + i)
    return res


def _hdf5_append_to_dataset(hdf_dataset, data):
    curr_size = hdf_dataset.shape[0]
    hdf_dataset.resize(curr_size + data.shape[0], 0)
    hdf_dataset[curr_size:] = data


class _dummy_shuffle():

    @staticmethod
    def shuffle(x):
        pass

    @staticmethod
    def permutation(x):
        return np.arange(x)

# ===========================================================================
# Main code
# ===========================================================================


class batch(object):

    """Batch object
    Parameters
    ----------
    key : str, list(str)
        list of dataset key in h5py file
    hdf : h5py.File
        a h5py.File or list of h5py.File
    arrays : numpy.ndarray, list(numpy.ndarray)
        if arrays is specified, ignore key and hdf
    chunk_size : int, 'auto'
        chunks size for creating auto-resize dataset in h5py, very important,
        affect the reading speed of dataset
    Note
    ----
    Chunk size should give a power of 2 to the number of samples, for example:
    shape=(1,500,120) => chunk_size=(8,500,120) or (16,500,120)
    Chunk size should be divisible by/for batch size, for example:
    batch_size = 128 => chunk_size=2,4,8,16,32,64,128,256,512,...
    The higher, the faster but consume more external storage and also more RAM
    when reading from dataset
    - Fastest reading speed for batch_size=128 is chunk_size=128
    """

    def __init__(self, key=None, hdf=None, arrays=None, chunk_size=16):
        super(batch, self).__init__()
        self._normalizer = lambda x: x
        self._chunk_size = chunk_size
        # hdf5 mode
        if arrays is None:
            if type(key) not in (tuple, list):
                key = [key]
            if type(hdf) not in (tuple, list):
                hdf = [hdf]
            if len(key) != len(hdf):
                raise ValueError('[key] and [hdf] must be equal size')

            self._key = key
            self._hdf = hdf
            self._data = []

            for key, hdf in zip(self._key, self._hdf):
                if key in hdf:
                    self._data.append(hdf[key])

            if len(self._data) > 0 and len(self._data) != len(self._hdf):
                raise ValueError('Not all [hdf] file contain given [key]')
            self._is_array_mode = False
        # arrays mode
        else:
            if type(arrays) not in (tuple, list):
                arrays = [arrays]
            self._data = arrays
            self._is_array_mode = True

    def _check(self, shape, dtype):
        # if not exist create initial dataset
        if len(self._data) == 0:
            for key, hdf in zip(self._key, self._hdf):
                if key not in hdf:
                    hdf.create_dataset(key, dtype=dtype,
                        chunks=_get_chunk_size(shape, self._chunk_size),
                        shape=(0,) + shape[1:], maxshape=(None, ) + shape[1:])
                self._data.append(hdf[key])

        # check shape match
        for d in self._data:
            if d.shape[1:] != shape[1:]:
                raise TypeError('Shapes not match ' + str(d.shape) + ' - ' + str(shape))

    # ==================== Properties ==================== #
    def _is_dataset_init(self):
        if len(self._data) == 0:
            raise RuntimeError('Dataset have not initialized yet!')

    @property
    def shape(self):
        self._is_dataset_init()
        s = sum([i.shape[0] for i in self._data])
        return (s,) + i.shape[1:]

    @property
    def dtype(self):
        self._is_dataset_init()
        return self._data[0].dtype

    @property
    def value(self):
        self._is_dataset_init()
        if self._is_array_mode:
            return np.concatenate([i for i in self._data], axis=0)
        return np.concatenate([i.value for i in self._data], axis=0)

    def set_normalizer(self, normalizer):
        '''
        Parameters
        ----------
        normalizer : callable
            a function(X)
        '''
        if normalizer is None:
            self._normalizer = lambda x: x
        else:
            self._normalizer = normalizer
        return self

    # ==================== Arithmetic ==================== #
    def sum2(self, axis=0):
        ''' sum(X^2) '''
        self._is_dataset_init()
        if self._is_array_mode:
            return np.power(self[:], 2).sum(axis)

        s = 0
        isInit = False
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                # for more stable precision
                s += np.sum(np.power(X, 2), 0)
            else:
                if not isInit:
                    s = [np.sum(np.power(X, 2), axis)]
                    isInit = True
                else:
                    s.append(np.sum(np.power(X, 2), axis))
        if isinstance(s, list):
            s = np.concatenate(s, axis=0)
        return s

    def sum(self, axis=0):
        ''' sum(X) '''
        self._is_dataset_init()
        if self._is_array_mode:
            return self[:].sum(axis)

        s = 0
        isInit = False
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                # for more stable precision
                s += np.sum(X, 0)
            else:
                if not isInit:
                    s = [np.sum(X, axis)]
                    isInit = True
                else:
                    s.append(np.sum(X, axis))
        if isinstance(s, list):
            s = np.concatenate(s, axis=0)
        return s

    def mean(self, axis=0):
        self._is_dataset_init()

        s = self.sum(axis)
        return s / self.shape[axis]

    def var(self, axis=0):
        self._is_dataset_init()
        if self._is_array_mode:
            return np.var(np.concatenate(self._data, 0), axis)

        v2 = 0
        v1 = 0
        isInit = False
        n = self.shape[axis]
        for X in self.iter(shuffle=False):
            X = X.astype(np.float64)
            if axis == 0:
                v2 += np.sum(np.power(X, 2), axis)
                v1 += np.sum(X, axis)
            else:
                if not isInit:
                    v2 = [np.sum(np.power(X, 2), axis)]
                    v1 = [np.sum(X, axis)]
                    isInit = True
                else:
                    v2.append(np.sum(np.power(X, 2), axis))
                    v1.append(np.sum(X, axis))
        if isinstance(v2, list):
            v2 = np.concatenate(v2, axis=0)
            v1 = np.concatenate(v1, axis=0)
        v = v2 - 1 / n * np.power(v1, 2)
        return v / n

    # ==================== manupilation ==================== #
    def append(self, other):
        if not isinstance(other, np.ndarray):
            raise TypeError('Append only support numpy ndarray')
        self._check(other.shape, other.dtype)
        if self._is_array_mode:
            self._data = [np.concatenate((i, other), 0) for i in self._data]
        else: # hdf5
            for d in self._data:
                _hdf5_append_to_dataset(d, other)
        return self

    def duplicate(self, other):
        self._is_dataset_init()
        if not isinstance(other, int):
            raise TypeError('Only duplicate by int factor')
        if len(self._data) == 0:
            raise TypeError("Data haven't initlized yet")

        if self._is_array_mode:
            self._data = [np.concatenate([i] * other, 0) for i in self._data]
        else: # hdf5
            for d in self._data:
                n = d.shape[0]
                batch_size = int(max(0.1 * n, 1))
                for i in xrange(other - 1):
                    copied = 0
                    while copied < n:
                        copy = d[copied: int(min(copied + batch_size, n))]
                        _hdf5_append_to_dataset(d, copy)
                        copied += batch_size
        return self

    def sample(self, size, seed=None, proportion=True):
        '''
        Parameters
        ----------
        proportion : bool
            will the portion of each dataset in the batch reserved the same
            as in original dataset
        '''
        if seed:
            np.random.seed(seed)

        all_size = [i.shape[0] for i in self._data]
        s = sum(all_size)
        if proportion:
            idx = [sorted(
                np.random.permutation(i)[:round(size * i / s)].tolist()
            )
                for i in all_size]
        else:
            size = int(size / len(all_size))
            idx = [sorted(np.random.permutation(i)[:size].tolist())
                   for i in all_size]
        return np.concatenate([i[j] for i, j in zip(self._data, idx)], 0)

    def _iter_fast(self, ds, batch_size, start=None, end=None,
            shuffle=True, seed=None):
        # craete random seed
        prng1 = None
        prng2 = _dummy_shuffle
        if shuffle:
            if seed is None:
                seed = get_random_magic_seed()
            prng1 = RandomState(seed)
            prng2 = RandomState(seed)

        batches = create_batch(ds.shape[0], batch_size, start, end, prng1)
        prng2.shuffle(batches)
        for i, j in batches:
            data = ds[i:j]
            yield self._normalizer(data[prng2.permutation(data.shape[0])])

    def _iter_slow(self, batch_size=128, start=None, end=None,
                   shuffle=True, seed=None, mode=0):
        # ====== Set random seed ====== #
        all_ds = self._data[:]
        prng1 = None
        prng2 = _dummy_shuffle
        if shuffle:
            if seed is None:
                seed = get_random_magic_seed()
            prng1 = RandomState(seed)
            prng2 = RandomState(seed)

        all_size = [i.shape[0] for i in all_ds]
        n_dataset = len(all_ds)

        # ====== Calculate batch_size ====== #
        if mode == 1: # equal
            s = sum(all_size)
            all_batch_size = [int(round(batch_size * i / s)) for i in all_size]
            for i in xrange(len(all_batch_size)):
                if all_batch_size[i] == 0: all_batch_size[i] += 1
            if sum(all_batch_size) > batch_size: # 0.5% -> round up, too much
                for i in xrange(len(all_batch_size)):
                    if all_batch_size[i] > 1:
                        all_batch_size[i] -= 1
                        break
            all_upsample = [None] * len(all_size)
        elif mode == 2 or mode == 3: # upsampling and downsampling
            maxsize = int(max(all_size)) if mode == 2 else int(min(all_size))
            all_batch_size = [int(batch_size / n_dataset) for i in xrange(n_dataset)]
            for i in xrange(batch_size - sum(all_batch_size)): # not enough
                all_batch_size[i] += 1
            all_upsample = [maxsize for i in xrange(n_dataset)]
        else: # sequential
            all_batch_size = [batch_size]
            all_upsample = [None]
            all_size = [sum(all_size)]
        # ====== Create all block and batches ====== #
        # [ ((idx1, batch1), (idx2, batch2), ...), # batch 1
        #   ((idx1, batch1), (idx2, batch2), ...), # batch 2
        #   ... ]
        all_block_batch = []
        # contain [block_batches1, block_batches2, ...]
        tmp_block_batch = []
        for n, batchsize, upsample in zip(all_size, all_batch_size, all_upsample):
            tmp_block_batch.append(
                create_batch(n, batchsize, start, end, prng1, upsample))
        # ====== Distribute block and batches ====== #
        if mode == 1 or mode == 2 or mode == 3:
            for i in zip_longest(*tmp_block_batch):
                all_block_batch.append([(k, v) for k, v in enumerate(i) if v is not None])
        else:
            all_size = [i.shape[0] for i in all_ds]
            all_idx = []
            for i, j in enumerate(all_size):
                all_idx += [(i, k) for k in xrange(j)] # (ds_idx, index)
            all_idx = [all_idx[i[0]:i[1]] for i in tmp_block_batch[0]]
            # complex algorithm to connecting the batch with different dataset
            for i in all_idx:
                tmp = []
                idx = i[0][0] # i[0][0]: ds_index
                start = i[0][1] # i[0][1]: index
                end = start
                for j in i[1:]: # detect change in index
                    if idx != j[0]:
                        tmp.append((idx, (start, end + 1)))
                        idx = j[0]
                        start = j[1]
                    end = j[1]
                tmp.append((idx, (start, end + 1)))
                all_block_batch.append(tmp)
        prng2.shuffle(all_block_batch)
        # print if you want debug
        # for _ in all_block_batch:
        #     for i, j in _:
        #         print('ds:', i, '  batch:', j)
        #     print('===== End =====')
        # ====== return iteration ====== #
        for _ in all_block_batch: # each _ is a block
            batches = np.concatenate(
                [all_ds[i][j[0]:j[1]] for i, j in _], axis=0)
            batches = batches[prng2.permutation(batches.shape[0])]
            yield self._normalizer(batches)

    def iter(self, batch_size=128, start=None, end=None,
        shuffle=True, seed=None, normalizer=None, mode=0):
        ''' Create iteration for all dataset contained in this _batch
        When [start] and [end] are given, it mean appying for each dataset
        If the amount of data between [start] and [end] < 1.0

        Parameters
        ----------
        batch_size : int, 'auto'
            size of each batch (data will be loaded in big block 8 times
            larger than this size)
        start : int, float(0.0-1.0)
            start point in dataset, will apply for all dataset
        end : int, float(0.0-1.0)
            end point in dataset, will apply for all dataset
        shuffle : bool, str
            wheather enable shuffle
        seed : int
        normalizer : callable, function
            funciton will be applied to each batch before return
        mode : 0, 1, 2
            0 - default, sequentially read each dataset
            1 - parallel read: proportionately for each dataset (e.g.
                batch_size=512, dataset1_size=1000, dataset2_size=500
                => ds1=341, ds2=170)
            2 - parallel read (upsampling): upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)
            3 - parallel read (downsampling): downsampling larger dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size <<
                dataset size)

        Returns
        -------
        return : generator
            generator generate batches of data

        Notes
        -----
        This method is thread-safe, as it uses private instance of RandomState.
        To create consistent permutation of shuffled dataset, you must:
         - both batch have the same number and order of dataset
         - using the same seed and mode when calling iter()
        Hint: small level of batch shuffle can be obtained by using normalizer
        function
        The only case that 2 [dnntoolkit.batch] have the same order is when
        mode=0 and shuffle=False, for example
        >>> X = batch(['X1','X2], [f1, f2])
        >>> y = batch('Y', f)
        >>> X.iter(512, mode=0, shuffle=False) have the same order with
            y.iter(512, mode=0, shuffle=False)
        '''
        self._is_dataset_init()
        if normalizer is not None:
            self.set_normalizer(normalizer)
        if batch_size == 'auto':
            batch_size = _auto_batch_size(self.shape)

        if len(self._data) == 1:
            return self._iter_fast(self._data[0], batch_size, start, end,
                                   shuffle, seed)
        else:
            return self._iter_slow(batch_size, start, end, shuffle, seed,
                                   mode)

    def iter_len(self, mode=0):
        '''This methods return estimated iteration length'''
        self._is_dataset_init()
        if mode == 2: #upsampling
            maxlen = max([i.shape[0] for i in self._data])
            return int(maxlen * len(self._data))
        if mode == 3: #downsampling
            minlen = min([i.shape[0] for i in self._data])
            return int(minlen * len(self._data))
        return len(self)

    def __len__(self):
        ''' This method return actual len by sum all shape[0] '''
        self._is_dataset_init()
        return sum([i.shape[0] for i in self._data])

    def __getitem__(self, key):
        self._is_dataset_init()
        if type(key) == tuple:
            return np.concatenate(
                [d[k] for k, d in zip(key, self._data) if k is not None],
                axis=0)
        return np.concatenate([i[key] for i in self._data], axis=0)

    def __setitem__(self, key, value):
        '''
        '''
        self._is_dataset_init()
        self._check(value.shape, value.dtype)
        if isinstance(key, slice):
            for d in self._data:
                d[key] = value
        elif isinstance(key, tuple):
            for k, d in zip(key, self._data):
                if k is not None:
                    d[k] = value

    def __str__(self):
        if len(self._data) == 0:
            return '<batch: None>'
        s = '<batch: '
        if self._is_array_mode: key = [''] * len(self._data)
        else: key = self._key
        for k, d in zip(key, self._data):
            s += '[%s,%s,%s]-' % (k, d.shape, d.dtype)
        s = s[:-1] + '>'
        return s


class dataset(OdinObject):

    '''
    dataset object to manage multiple hdf5 file

    Note
    ----
    dataset['X1', 'X2']: will search for 'X1' in all given files, then 'X2',
    then 'X3' ...
    iter(batch_size, start, end, shuffle, seed, normalizer, mode)
        mode : 0, 1, 2
            0 - default, read one by one each dataset
            1 - equally read each dataset, upsampling smaller dataset
                (e.g. batch_size=512, there are 5 dataset => each dataset
                102 samples) (only work if batch size << dataset size)
            2 - proportionately read each dataset (e.g. batch_size=512,
                dataset1_size=1000, dataset2_size=500 => ds1=341, ds2=170)
    '''

    def __init__(self, path, mode='r', chunk_size=16):
        super(dataset, self).__init__()
        self._mode = mode
        if type(path) not in (list, tuple):
            path = [path]

        self._hdf = [h5py.File(p, mode=mode) for p in path]
        self._write_mode = None
        self._isclose = False
        self.set_chunk_size(chunk_size)

        self._update_indexing()

    def _update_indexing(self):
        # map: ("ds_name1","ds_name2",...): <_batch object>
        self._batchmap = {}
        # all dataset have dtype('O') type will be return directly
        self._object = defaultdict(list)
        # obj_type = np.dtype('O')
        # map: "ds_name":[hdf1, hdf2, ...]
        self._index = defaultdict(list)
        for f in self._hdf:
            ds = _hdf5_get_all_dataset(f)
            for i in ds:
                tmp = f[i]
                if len(tmp.shape) == 0:
                    self._object[i].append(f)
                else:
                    self._index[i].append(f)

    def get_path(self):
        return [i.filename for i in self._hdf]

    # ==================== Set and get ==================== #
    def set_chunk_size(self, size):
        '''
        size: int, 'auto' (default: 16)
            - 'auto' chunk size
            - int is the number of sample stored in each chunk
            when converted to KiB, should be between 10 KiB and 1 MiB,
            larger for larger datasets
        Note
        ----
        - Chunk size should give a power of 2 to the number of samples,
        for example: shape=(1,500,120) => chunk_size=(8,500,120) or (16,500,120)
        - Chunk size should be divisible by/for batch size, for example:
        batch_size = 128 => chunk_size=2,4,8,16,32,64,128,256,512,...
        - The higher, the faster but consume more external storage and also more
        RAM when reading from dataset
        - Fastest reading speed for batch_size=128 is chunk_size=128
        - 'auto' is never a good choice for big dataset
        '''
        if isinstance(size, str):
            self._chunk_size = 'auto'
        else:
            self._chunk_size = int(size)

    def set_write(self, mode):
        '''
        Parameters
        ----------
        mode : 'all', 'last', int(index), slice
            specify which hdf files will be used for write
        '''
        self._write_mode = mode

    def is_close(self):
        return self._isclose

    def get_all_dataset(self, fileter_func=None, path='/'):
        ''' Get all dataset contained in the hdf5 file.

        Parameters
        ----------
        filter_func : function
            filter function applied on the name of dataset to find
            appropriate dataset
        path : str
            path start searching for dataset

        Returns
        -------
        return : list(str)
            names of all dataset
        '''
        all_dataset = []
        for i in self._hdf:
            all_dataset += _hdf5_get_all_dataset(i, fileter_func, path)
        return all_dataset

    # ==================== Main ==================== #
    def _get_write_hdf(self):
        '''Alawys return list of hdf5 files'''
        # ====== Only 1 file ====== #
        if len(self._hdf) == 1:
            return self._hdf

        # ====== Multple files ====== #
        if self._write_mode is None:
            self.log('Have not set write mode, default is [last]', 30)
            self._write_mode = 'last'

        if self._write_mode == 'last':
            return [self._hdf[-1]]
        elif self._write_mode == 'all':
            return self._hdf
        elif isinstance(self._write_mode, str):
            return [h for h in self._hdf if h.filename == self._write_mode]
        elif type(self._write_mode) in (tuple, list):
            if isinstance(self._write_mode[0], int):
                return [self._hdf[i] for i in self._write_mode]
            else: # search all file with given name
                hdf = []
                for k in self._write_mode:
                    for h in self._hdf:
                        if h.filename == k:
                            hdf.append(h)
                            break
                return hdf
        else: # int or slice index
            hdf = self._hdf[self._write_mode]
            if type(hdf) not in (tuple, list):
                hdf = [hdf]
            return hdf

    def __getitem__(self, key):
        ''' Logic of this function:
         - Object with no shape is returned directly, even though
         key is mixture of object and array, only return objects
         - Find as much as possible all hdf contain the key
         Example
         -------
         hdf1: a, b, c
         hdf2: a, b
         hdf3: a
         ==> key = a return [hdf1, hdf2, hdf3]
         ==> key = (a, b) return [hdf1, hdf2, hdf3, hdf1, hdf2]
        '''
        if type(key) not in (tuple, list):
            key = [key]

        # ====== Return object is priority ====== #
        ret = []
        for k in key:
            if k in self._object:
                ret += [i[k].value for i in self._object[k]]
        if len(ret) == 1:
            return ret[0]
        elif len(ret) > 0:
            return ret

        # ====== Return _batch ====== #
        keys = []
        for k in key:
            hdf = self._index[k]
            if len(hdf) == 0 and self._mode != 'r': # write mode activated
                hdf = self._get_write_hdf()
            ret += hdf
            keys += [k] * len(hdf)
        idx = tuple(keys + ret)

        if idx in self._batchmap:
            return self._batchmap[idx]
        b = batch(keys, ret, chunk_size=self._chunk_size)
        self._batchmap[idx] = b
        return b

    def __setitem__(self, key, value):
        ''' Logic of this function:
         - If mode is 'r': NO write => error
         - if object, write the object directly
         - mode=all: write to all hdf
         - mode=last: write to the last one
         - mode=slice: write to selected
        '''
        # check input
        if self._mode == 'r':
            raise RuntimeError('No write is allowed in read mode')
        if type(key) not in (tuple, list):
            key = [key]

        # set str value directly
        if isinstance(value, (tuple, list, str)) or \
            (hasattr(value, 'shape') and len(value.shape) == 0):
            hdf = self._get_write_hdf()
            for i in hdf:
                for j in key:
                    i[j] = value
                    self._object[j].append(i)
        else: # array
            if self._chunk_size == 'auto':
                self.log('Chunk size auto is not recommended for big dataset', 30)
            shape = value.shape
            # find appropriate key
            hdf = self._get_write_hdf()
            for k in key: # each key
                for h in hdf: # do the same for all writable hdf
                    if k in h: # already in the hdf files
                        h[k][:] = value
                    else: # always create reszie-able dataset
                        h.create_dataset(k, data=value,
                            dtype=value.dtype,
                            chunks=_get_chunk_size(shape, self._chunk_size),
                            shape=shape, maxshape=(None,) + shape[1:])
                        self._index[k].append(h)

    def __contains__(self, key):
        r = False
        for hdf in self._hdf:
            if key in hdf:
                r += 1
        return r

    def close(self):
        try:
            for i in self._hdf:
                i.close()
        except:
            pass
        self._isclose = True

    def __del__(self):
        try:
            for i in self._hdf:
                i.close()
            del self._hdf
        except:
            pass
        self._isclose = True

    def __str__(self):
        self._update_indexing()
        s = 'Dataset contains: %d (files)' % len(self._hdf) + '\n'
        if self._isclose:
            s += '******** Closed ********\n'
        else:
            s += '******** Array ********\n'
            all_data = self._index.keys() # faster
            for i in all_data:
                all_hdf = self._index[i]
                for j in all_hdf:
                    s += ' - name:%-13s  shape:%-18s  dtype:%-8s  hdf:%s' % \
                        (i, j[i].shape, j[i].dtype, j.filename) + '\n'
            s += '******** Objects ********\n'
            all_data = self._object.keys() # faster
            for i in all_data:
                all_hdf = self._object[i]
                for j in all_hdf:
                    s += ' - name:%-13s  shape:%-18s  dtype:%-8s  hdf:%s' % \
                        (i, j[i].shape, j[i].dtype, j.filename) + '\n'
        return s[:-1]

    # ==================== Static loading ==================== #
    @staticmethod
    def load_mnist(path='https://s3.amazonaws.com/ai-datasets/mnist.h5'):
        '''
        path : str
            local path or url to hdf5 datafile
        '''
        datapath = get_file('mnist.h5', path)
        logger.info('Loading data from: %s' % datapath)
        try:
            ds = dataset(datapath, mode='r')
        except:
            if os.path.exists(datapath):
                os.remove(datapath)
            datapath = get_file('mnist.h5', path)
            ds = dataset(datapath, mode='r')
        return ds

    def load_imdb(path):
        pass

    def load_reuters(path):
        pass
