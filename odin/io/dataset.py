from __future__ import print_function, division, absolute_import

import os
import numpy as np
from math import ceil

from ..base import OdinObject
from .ie import get_file
from ..tensor import get_random_magic_seed
from .. import logger
from .data import MmapData, Hdf5Data, load_data, open_hdf5

from six.moves import zip_longest, zip, range
from collections import OrderedDict
from itertools import chain

__all__ = [
    'DataIterator',
    'dataset'
]


# ===========================================================================
# data iterator
# ===========================================================================
def _approximate_continuos_by_discrete(distribution):
    '''original distribution: [ 0.47619048  0.38095238  0.14285714]
       best approximated: [ 5.  4.  2.]
    '''
    distribution = 1 - distribution
    x = np.round(1 / distribution)
    x = np.where(distribution == 0, 0, x)
    return x.astype(int)


class DataIterator(object):

    ''' Vertically merge several data object for iteration

        mode : 0, 1, 2, 3, tuple or list
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
            tuple or list - distribution of each data
    '''

    def __init__(self, data, batch_size=256, shuffle=True, seed=None):
        if not isinstance(data, (tuple, list)):
            data = (data,)
        if any(not isinstance(i, (MmapData, Hdf5Data)) for i in data):
            raise ValueError('data must be instance of MmapData or Hdf5Data, '
                             'but given data have types: {}'
                             ''.format(map(lambda x: str(type(x)).split("'")[1],
                                          data)))
        shape = data[0].shape[1:]
        if any(i.shape[1:] != shape for i in data):
            raise ValueError('all data must have the same trial dimension, but'
                             'given shape of all data as following: {}'
                             ''.format([i.shape for i in data]))
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.RandomState(seed)

        self._start = 0.
        self._end = 1.

        self._sequential = False
        self._distribution = [1.] * len(data)

    # ==================== properties ==================== #
    def __len__(self):
        return sum(i * j.shape[0]
                   for i, j in zip(self._distribution, self._data))

    @property
    def data(self):
        return self._data

    @property
    def distribution(self):
        return self._distribution

    # ==================== batch configuration ==================== #
    def set_mode(self, sequential=None, distribution=None):
        if sequential is not None:
            self._sequential = sequential
        if distribution is not None:
            # upsampling or downsampling
            if isinstance(distribution, str):
                distribution = distribution.lower()
                if 'up' in distribution or 'over' in distribution:
                    n = max(i.shape[0] for i in self._data)
                elif 'down' in distribution or 'under' in distribution:
                    n = min(i.shape[0] for i in self._data)
                else:
                    raise ValueError("Only upsampling (keyword: up, over) "
                                     "or undersampling (keyword: down, under) "
                                     "are supported.")
                self._distribution = [n / i.shape[0] for i in self._data]
            # real values distribution
            elif isinstance(distribution, (tuple, list)):
                if len(distribution) != len(self._data):
                    raise ValueError('length of given distribution must equal '
                                     'to number of data in the iterator, but '
                                     'len_data={} != len_distribution={}'
                                     ''.format(len(self._data), len(self._distribution)))
                self._distribution = distribution
            # all the same value
            elif isinstance(distribution, float):
                self._distribution = [distribution] * len(self._data)
        return self

    def set_range(self, start, end):
        if start < 0 or end < 0:
            raise ValueError('start and end must > 0, but start={} and end={}'
                             ''.format(start, end))
        self._start = start
        self._end = end
        return self

    def set_batch(self, batch_size=None, shuffle=None, seed=None):
        if batch_size is not None:
            self._batch_size = batch_size
        if shuffle is not None:
            self._shuffle = shuffle
        if seed is not None:
            self._rng.seed(seed)
        return self

    # ==================== main logic of batch iterator ==================== #
    def _seed(self):
        if self._shuffle:
            return self._rng.randint(10e8)
        return None

    def __iter__(self):
        # ====== easy access many private variables ====== #
        sequential = self._sequential
        start, end = self._start, self._end
        batch_size = self._batch_size
        data = np.asarray(self._data)
        distribution = np.asarray(self._distribution)
        if self._shuffle: # shuffle order of data (good for sequential mode)
            idx = self._rng.permutation(len(data))
            data = data[idx]
            distribution = distribution[idx]
        shape = [i.shape[0] for i in data]
        # ====== prepare distribution information ====== #
        # number of sample should be traversed
        n = np.asarray([i * j for i, j in zip(distribution, shape)])
        n = np.round(n).astype(int)
        # normalize the distribution (base on new sample n of each data)
        distribution = n / n.sum()
        distribution = _approximate_continuos_by_discrete(distribution)
        # somehow heuristic, rescale distribution to get more benifit from cache
        if distribution.sum() <= len(data):
            distribution = distribution * 3
        # distribution now the actual batch size of each data
        distribution = (batch_size * distribution).astype(int)
        assert distribution.sum() % batch_size == 0, 'wrong distribution size!'
        # predefined (start,end) pair of each batch (e.g (0,256), (256,512))
        idx = list(range(0, batch_size + distribution.sum(), batch_size))
        idx = list(zip(idx, idx[1:]))
        # ==================== optimized parallel code ==================== #
        if not sequential:
            # first iterators
            it = [iter(dat.set_batch(bs, self._seed(), start, end))
                  for bs, dat in zip(distribution, data)]
            # iterator
            while sum(n) > 0:
                data = []
                for i, x in enumerate(it):
                    if n[i] <= 0:
                        continue
                    try:
                        x = x.next()[:n[i]]
                        n[i] -= x.shape[0]
                        data.append(x)
                    except StopIteration: # one iterator stopped
                        it[i] = iter(data[i].set_batch(
                            distribution[i], self._seed(), start, end))
                        x = it[i].next()[:n[i]]
                        n[i] -= x.shape[0]
                        data.append(x)
                # got final data
                data = np.vstack(data)
                if self._shuffle:
                    # no idea why random permutation is much faster than shuffle
                    data = data[self._rng.permutation(data.shape[0])]
                    # self._rng.shuffle(data)
                for start, end in idx[:int(ceil(data.shape[0] / batch_size))]:
                    yield data[start:end]
        # ==================== optimized sequential code ==================== #
        else:
            # first iterators
            batch_size = distribution.sum()
            it = [iter(dat.set_batch(batch_size, self._seed(), start, end))
                  for dat in data]
            current_data = 0
            # iterator
            while sum(n) > 0:
                if n[current_data] <= 0:
                    current_data += 1
                try:
                    data = it[current_data].next()[:n[current_data]]
                    n[current_data] -= data.shape[0]
                except StopIteration: # one iterator stopped
                    it[current_data] = iter(data[current_data].set_batch(
                        batch_size, self._seed(), start, end))
                    data = it[current_data].next()[:n[current_data]]
                    n[current_data] -= data.shape[0]
                if self._shuffle:
                    data = data[self._rng.permutation(data.shape[0])]
                for i, j in idx[:int(ceil(data.shape[0] / self._batch_size))]:
                    yield data[i:j]


# ===========================================================================
# dataset
# ===========================================================================
class dataset(OdinObject):

    def __init__(self, path):
        super(OdinObject, self).__init__()
        path = os.path.abspath(path)
        self._data_map = OrderedDict()

        if not os.path.exists(path):
            os.mkdir(path)
        elif not os.path.isdir(path):
            raise ValueError('Dataset path must be folder.')

        files = os.listdir(path)
        for f in files:
            f = os.path.join(path, f)
            data = load_data(f)
            if data is None:
                continue
            for d in data:
                # shape[1:], because first dimension can be resize afterward
                key = (d.name, d.dtype, d.shape[1:])
                if key in self._data_map:
                    raise ValueError('Found duplicated data: name={} in '
                                     '{}, registered data: (name={}; path={})'
                                     ''.format(d.name, d.path,
                                               self._data_map[key].name,
                                               self._data_map[key].path))
                else:
                    self._data_map[key] = d

        self._path = path
        self._name = os.path.basename(path)
        if len(self._name) == 1:
            self._name = os.path.basename(os.path.abspath(path))
        self._default_hdf5 = self.name + '_default.h5'

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._path

    @property
    def archive_path(self):
        return os.path.join(self._path, self._name + '.zip')

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        ''' return size in MegaByte'''
        size_bytes = 0
        for i in self._data_map.values():
            size = np.dtype(i.dtype).itemsize
            n = np.prod(i.shape)
            size_bytes += size * n
        return size_bytes / 1024. / 1024.

    # ==================== manipulate data ==================== #
    def get_data(self, name, dtype=None, shape=None, datatype='mmap'):
        datatype = datatype.lower()
        if datatype not in ['mmap', 'memmap', 'mem', 'hdf5', 'h5', 'hdf']:
            raise ValueError("No support for data type: {}, following formats "
                             " are supported: 'mmap', 'memmap', 'hdf5', 'h5'"
                             "".format(datatype))

        return_data = None
        # ====== find defined data ====== #
        for k in self._data_map.keys():
            _name, _dtype, _shape = k
            if name == _name:
                if dtype is not None and np.dtype(_dtype) != np.dtype(dtype):
                    continue
                if shape is not None and shape[1:] != _shape:
                    continue
                return_data = self._data_map[k]
                break
        # ====== auto create new data, if cannot find any match ====== #
        if return_data is None and dtype is not None and shape is not None:
            if datatype in ['mmap', 'memmap', 'mem']:
                return_data = MmapData(os.path.join(self.path, name),
                    dtype=dtype, shape=shape, mode='w+', override=True)
            else:
                f = open_hdf5(os.path.join(self.path, self._default_hdf5), mode='a')
                return_data = Hdf5Data(name, f, dtype=dtype, shape=shape)
            key = (return_data.name, return_data.dtype, return_data.shape)
            self._data_map[key] = return_data
        if return_data is None:
            raise ValueError('Cannot find or create data with name={}, dtype={} '
                             'shape={}, and datatype={}'
                             ''.format(name, dtype, shape, datatype))
        return return_data

    def create_iter(self, names,
        batch_size=256, shuffle=True, seed=None, start=0., end=1., mode=0):
        if seed is None:
            seed = get_random_magic_seed()

    def archive(self):
        from zipfile import ZipFile, ZIP_DEFLATED
        path = self.archive_path
        zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

        files = []
        for i in self._data_map.values():
            files.append(i.path)
        files = set(files)

        maxlen = max([len(os.path.basename(i)) for i in files])
        for i, f in enumerate(files):
            zfile.write(f, os.path.basename(f))
            logger.progress(i + 1, len(files),
            title=('Archiving: %-' + str(maxlen) + 's') % os.path.basename(f))
        # print()
        zfile.close()
        return path

    def flush(self):
        for i in self._data_map.values():
            i.flush()

    # ==================== Some info ==================== #
    def __str__(self):
        s = ['====== Dataset:%s ======' % self.path]
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        longest_file = 0
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
            longest_file = max(len(str(data.path)), longest_file)
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  '
                      'file:%-' + str(longest_file) + 's')
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape
            s.append(format_str % (name, dtype, shape, data.path))
        return '\n'.join(s)

    # ==================== Static loading ==================== #
    @staticmethod
    def load(path, extract_path='.'):
        if '.zip' not in path or not os.path.exists(path):
            raise ValueError('only compressed ZipFile of dataset folder is '
                             'accepted, is_zip_file={}, is_file_exist={}'
                             '.'.format('.zip' in path, os.path.exists(path)))
        from zipfile import ZipFile, ZIP_DEFLATED
        try:
            zfile = ZipFile(path, mode='r', compression=ZIP_DEFLATED)
            # validate extract_path
            if os.path.isfile(extract_path):
                raise ValueError('Extract path must be path folder, but path'
                                 '={} is a file'.format(extract_path))
            if not os.path.exists(extract_path):
                os.mkdir(extract_path)
            extract_path = os.path.join(extract_path,
                                        os.path.basename(path).replace('.zip', ''))
            zfile.extractall(path=extract_path)
            return dataset(extract_path)
        except IOError, e:
            raise IOError('Error loading archived dataset, path:{}, error:{}'
                          '.'.format(path, e))
        return None

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
