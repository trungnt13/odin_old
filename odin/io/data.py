# ===========================================================================
# Class handle extreme large numpy ndarray
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import numpy as np

import re
from .. import tensor as T
from ..decorators import autoattr, cache, typecheck
from ..utils import queue
from six.moves import range, zip

from collections import defaultdict


__all__ = [
    'load_data',
    'open_hdf5',
    'resize_memmap',
    'MmapData',
    'Hdf5Data'
]


# ===========================================================================
# Helper function
# ===========================================================================
def resize_memmap(memmap, shape):
    if not isinstance(memmap, np.core.memmap):
        raise ValueError('Object must be instance of numpy memmap')

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    if any(i != j for i, j in zip(shape[1:], memmap.shape[1:])):
        raise ValueError('Resize only support the first dimension, but '
                         '{} != {}'.format(shape[1:], memmap.shape[1:]))
    if shape[0] < memmap.shape[0]:
        raise ValueError('Only support extend memmap, and do not shrink the memory')
    elif shape[0] == memmap.shape[0]:
        return memmap
    memmap.flush()
    memmap = np.memmap(memmap.filename,
                       dtype=memmap.dtype,
                       mode='r+',
                       shape=(shape[0],) + tuple(memmap.shape[1:]))
    return memmap


def _get_chunk_size(shape, size):
    if isinstance(size, int):
        return (2**int(np.ceil(np.log2(size))),) + shape[1:]
    elif size is None:
        return False
    return True


def load_data(file):
    ''' Always return a list of Data '''
    match = MmapData.PATTERN.search(file)
    if not os.path.isfile(file): # only support data file
        return None

    if match is not None:
        try:
            mmap = MmapData(file, mode='r+', dtype=None, shape=None)
            return [mmap]
        except Exception, e:
            raise ValueError('Error loading memmap data, error:{}, file:{}'
                             ''.format(e, file))
    elif any(i in file for i in Hdf5Data.SUPPORT_EXT):
        try:
            f = open_hdf5(file, mode='r')
            ds = get_all_hdf_dataset(f)
            return [Hdf5Data(i, f) for i in ds]
        except Exception, e:
            raise ValueError('Error loading hdf5 data, error:{}, file:{} '
                             ''.format(e, file))
    return None


# ===========================================================================
# Memmap Data object
# ===========================================================================
class MmapData(object):

    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  Numpy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Delete the memmap instance to close.

    Parameters
    ----------
    filename : str or file-like object
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r+'.
    shape : tuple, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.
    """

    # name.float32.(8,12)
    PATTERN = re.compile('[a-zA-Z0-9]*\.[a-zA-Z]*\d{1,2}.\(\d+(,\d+)*\)')
    NAME_PATTERN = re.compile('[a-zA-Z0-9]*')
    SHAPE_PATTERN = re.compile('\.\(\d+(,\d+)*\)')
    DTYPE_PATTERN = re.compile('\.[a-zA-Z]*\d{1,2}')

    def __init__(self, path, mode='r+', dtype='float32', shape=None,
        override=True):
        # validate path
        name = os.path.basename(path)
        path = path.replace(name, '')
        path = path if len(path) > 0 else '.'
        if name[0] == '.':
            name = name[1:]

        name_match = self.NAME_PATTERN.search(name)
        shape_match = self.SHAPE_PATTERN.search(name)
        dtype_match = self.DTYPE_PATTERN.search(name)
        if shape_match is not None:
            shape = name[shape_match.start():shape_match.end()][1:]
            shape = eval(shape)
        if dtype_match is not None:
            dtype = name[dtype_match.start():dtype_match.end()][1:]
            dtype = np.dtype(dtype)
        if name_match is not None:
            name = name[name_match.start():name_match.end()]
        else:
            raise ValueError('Cannot find name of memmap from given path')

        if dtype is None:
            raise ValueError('Must specify dtype at the beginning, either in '
                             'the path or dtype argument.')
        if shape is not None:
            shape = tuple([1 if i is None else i for i in shape])

        # check mode
        if 'r' in mode or 'c' in mode:
            files = os.listdir(path)
            for f in files:
                match = self.PATTERN.match(f)
                if match is not None:
                    _name, _dtype, _shape = f.split('.')
                    _shape = eval(_shape)
                    if name == _name and dtype == _dtype:
                        if shape is None or shape[1:] == _shape[1:]:
                            shape = _shape
            mmap_path = os.path.join(path, self._info_to_name(name, shape, dtype))
            if not os.path.exists(mmap_path):
                raise ValueError('File not exist, given path:' + mmap_path)
        elif 'w' in mode:
            if shape is None:
                raise ValueError('dtype and shape must be specified in write '
                                 'mode, but shape={} and dtype={}'
                                 ''.format(shape, dtype))
            # the main issue is 2 files with the same name can have different
            # shape and dtype (this cannot happen)
            files = os.listdir(path)
            for f in files:
                if self.PATTERN.match(f) is not None:
                    _name, _dtype, _shape = f.split('.')
                    _shape = eval(_shape)
                    if _name == name and _dtype == dtype and shape[1:] == _shape[1:]:
                        if not override:
                            raise ValueError("memmap with the same name={} and dtype={} "
                                            "but with different shape already existed "
                                            "with name={}".format(name, dtype, f))
                        else: # remove old file to override
                            try:
                                os.remove(f)
                            except:
                                pass
            mmap_path = os.path.join(path, self._info_to_name(name, shape, dtype))
        else:
            raise ValueError('No support for given mode:' + str(mode))

        # store variables
        self._mmap = np.memmap(mmap_path, dtype=dtype, shape=shape, mode=mode)
        self._name = name.split('.')[0]
        self._path = path
        self._mode = mode

        # batch information
        self._batch_size = 256
        self._start = 0.
        self._end = 1.
        self._rng = np.random.RandomState()
        self._seed = None
        self._status = 0 # flag show that array valued changed

    def _name_to_info(self, name):
        shape = eval(name.split('.')[1])
        dtype = name.split('.')[2]
        return shape, dtype

    def _info_to_name(self, name, shape, dtype):
        return '.'.join([name,
                        str(dtype),
                        '(' + ','.join([str(i) for i in shape]) + ')'])

    # ==================== properties ==================== #
    @property
    def shape(self):
        return self._mmap.shape

    @property
    def dtype(self):
        return self._mmap.dtype

    @property
    def array(self):
        return np.asarray(self._mmap)

    def tolist(self):
        return self._mmap.tolist()

    @property
    def mmap(self):
        return self._mmap

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def path(self):
        return self._mmap.filename

    @property
    def name(self):
        return self._name

    def set_batch(self, batch_size=None, seed=None, start=None, end=None):
        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        self._seed = seed
        if start is not None and start > 0. - 1e-12:
            self._start = start
        if end is not None and end > 0. - 1e-12:
            self._end = end
        return self

    @autoattr(_status=lambda x: x + 1)
    def append(self, *arrays):
        accepted_arrays = []
        new_size = 0
        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == self.shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        old_size = self.shape[0]
        self.resize(old_size + new_size) # resize only once will be faster
        for a in accepted_arrays:
            self._mmap[old_size:old_size + a.shape[0]] = a
            old_size = old_size + a.shape[0]
        return self

    @autoattr(_status=lambda x: x + 1)
    def prepend(self, *arrays):
        accepted_arrays = []
        new_size = 0
        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == self.shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        if new_size > self.shape[0]:
            self.resize(new_size) # resize only once will be faster
        size = 0
        for a in accepted_arrays:
            self._mmap[size:size + a.shape[0]] = a
            size = size + a.shape[0]
        return self

    # ==================== High-level operator ==================== #
    @cache('_status')
    def sum(self, axis=0):
        return self._mmap.sum(axis)

    @cache('_status')
    def cumsum(self, axis=None):
        return self._mmap.cumsum(axis)

    @cache('_status')
    def sum2(self, axis=0):
        return self._mmap.__pow__(2).sum(axis)

    @cache('_status')
    def pow(self, y):
        return self._mmap.__pow__(y)

    @cache('_status')
    def min(self, axis=None):
        return self._mmap.min(axis)

    @cache('_status')
    def argmin(self, axis=None):
        return self._mmap.argmin(axis)

    @cache('_status')
    def max(self, axis=None):
        return self._mmap.max(axis)

    @cache('_status')
    def argmax(self, axis=None):
        return self._mmap.argmax(axis)

    @cache('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self.shape[i] for i in axis])
        return sum1 / n

    @cache('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._mmap -= mean
        self._mmap /= std
        return self

    # ==================== Special operators ==================== #
    def __add__(self, y):
        return self._mmap.__add__(y)

    def __sub__(self, y):
        return self._mmap.__sub__(y)

    def __mul__(self, y):
        return self._mmap.__mul__(y)

    def __div__(self, y):
        return self._mmap.__div__(y)

    def __floordiv__(self, y):
        return self._mmap.__floordiv__(y)

    def __pow__(self, y):
        return self._mmap.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._mmap.__iadd__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._mmap.__isub__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._mmap.__imul__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._mmap.__idiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._mmap.__ifloordiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        return self._mmap.__ipow__(y)

    def __neg__(self):
        self._mmap.__neg__()
        return self

    def __pos__(self):
        self._mmap.__pos__()
        return self

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        return self._mmap.__getitem__(y)

    @autoattr(_status=lambda x: x + 1)
    def __setitem__(self, x, y):
        return self._mmap.__setitem__(x, y)

    # ==================== iteration ==================== #
    def __iter__(self):
        batch_size = self._batch_size
        seed = self._seed
        rng = self._rng

        # custom batch_size
        start = (int(self._start * self.shape[0]) if self._start < 1. + 1e-12
                 else int(self._start))
        end = (int(self._end * self.shape[0]) if self._end < 1. + 1e-12
               else int(self._end))
        if start > self.shape[0] or end > self.shape[0]:
            raise ValueError('start={} or end={} excess data_size={}'
                             ''.format(start, end, self.shape[0]))

        idx = list(range(start, end, batch_size))
        if idx[-1] < end:
            idx.append(end)
        idx = list(zip(idx, idx[1:]))
        if seed is not None:
            rng.seed(seed)
            rng.shuffle(idx)
            self._seed = None

        for i in idx:
            start, end = i
            yield self._mmap[start:end]

    # ==================== Strings ==================== #
    def __str__(self):
        return self._mmap.__str__()

    def __repr__(self):
        return self._mmap.__repr__()

    # ==================== Save ==================== #
    def resize(self, shape):
        mode = self._mode
        mmap = self._mmap
        if mode == 'r' or mode == 'c':
            return

        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], mmap.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], mmap.shape[1:]))
        if shape[0] < mmap.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self._mmap.shape[0]:
            return self
        mmap.flush()
        # resize by create new memmap and also rename old file
        shape = (shape[0],) + tuple(mmap.shape[1:])
        new_name = os.path.join(os.path.dirname(self.path),
                                self._info_to_name(self.name, shape, mmap.dtype))
        os.rename(mmap.filename, new_name)
        self._mmap = np.memmap(new_name,
                               dtype=mmap.dtype,
                               mode='r+',
                               shape=shape)
        return self

    def flush(self):
        mode = self._mode
        if mode == 'r' or mode == 'c':
            return

        old_path = self._mmap.filename
        new_path = os.path.join(os.path.dirname(self.path),
                    self._info_to_name(self.name, self.shape, self.dtype))
        self._mmap.flush()
        if old_path != new_path:
            del self._mmap
            os.rename(old_path, new_path)
            if 'w' in mode:
                mode = 'r+'
            self._mmap = np.memmap(new_path, mode=mode)


# ===========================================================================
# Hdf5 Data object
# ===========================================================================
try:
    import h5py
except:
    pass
_HDF5 = {}


def get_all_hdf_dataset(hdf, fileter_func=None, path='/'):
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


def _validate_operate_axis(axis):
    ''' as we iterate over first dimension, it is prerequisite to
    have 0 in the axis of operator
    '''
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    axis = tuple(int(i) for i in axis)
    if 0 not in axis:
        raise ValueError('Expect 0 in the operating axis because we always'
                         ' iterate data over the first dimension.')
    return axis


def open_hdf5(path, mode='r'):
    '''
    Note
    ----
    If given file already open in read mode, mode = 'w' will cause error
    (this is good error and you should avoid this situation)

    '''
    if 'r' in mode:
        mode = 'r+'

    key = (path, mode)
    if key in _HDF5:
        f = _HDF5[key]
        if 'Closed' in str(f):
            f = h5py.File(path, mode=mode)
    else:
        f = h5py.File(path, mode=mode)
    _HDF5[key] = f
    return f


class Hdf5Data(object):

    SUPPORT_EXT = ['.h5', '.hdf', '.dat', '.hdf5']

    def __init__(self, dataset, hdf=None, dtype='float32', shape=None,
        chunk_size=32):
        self._chunk_size = chunk_size
        if isinstance(hdf, str):
            hdf = open_hdf5(hdf, mode='a')
        if hdf is None and not isinstance(dataset, h5py.Dataset):
            raise ValueError('Cannot initialize dataset without hdf file')

        if isinstance(dataset, h5py.Dataset):
            self._dataset = dataset
            self._hdf = dataset.file
        else:
            if dataset not in hdf: # not created dataset
                if dtype is None or shape is None:
                    raise ValueError('dtype and shape must be specified if '
                                     'dataset has not created in hdf5 file.')
                shape = tuple([0 if i is None else i for i in shape])
                hdf.create_dataset(dataset, dtype=dtype,
                    chunks=_get_chunk_size(shape, chunk_size),
                    shape=shape, maxshape=(None, ) + shape[1:])

            self._dataset = hdf[dataset]
            if shape is not None and self._dataset.shape[1:] != shape[1:]:
                raise ValueError('Shape mismatch between predefined dataset '
                                 'and given shape, {} != {}'
                                 ''.format(shape, self._dataset.shape))
            self._hdf = hdf

        self._batch_size = 256
        self._start = 0.
        self._end = 1.
        self._rng = np.random.RandomState()
        self._seed = None
        self._status = 0 # flag show that array valued changed

    # ==================== properties ==================== #
    @property
    def shape(self):
        return self._dataset.shape

    @property
    def dtype(self):
        return self._dataset.dtype

    @property
    def array(self):
        return self._dataset[:]

    def tolist(self):
        return self._dataset[:].tolist()

    @property
    def dataset(self):
        return self._dataset

    @property
    def hdf5(self):
        return self._hdf

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def path(self):
        return self._hdf.filename

    @property
    def name(self):
        _ = self._dataset.name
        if _[0] == '/':
            _ = _[1:]
        return _

    def set_batch(self, batch_size=None, seed=None, start=None, end=None):
        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        self._seed = seed
        if start is not None and start > 0. - 1e-12:
            self._start = start
        if end is not None and end > 0. - 1e-12:
            self._end = end
        return self

    @autoattr(_status=lambda x: x + 1)
    def append(self, *arrays):
        accepted_arrays = []
        new_size = 0
        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == self.shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        old_size = self.shape[0]
        self.resize(old_size + new_size) # resize only once will be faster
        for a in accepted_arrays:
            self._dataset[old_size:old_size + a.shape[0]] = a
            old_size = old_size + a.shape[0]
        return self

    @autoattr(_status=lambda x: x + 1)
    def prepend(self, *arrays):
        accepted_arrays = []
        new_size = 0
        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == self.shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        if new_size > self.shape[0]:
            self.resize(new_size) # resize only once will be faster
        size = 0
        for a in accepted_arrays:
            self._dataset[size:size + a.shape[0]] = a
            size = size + a.shape[0]
        return self

    # ==================== High-level operator ==================== #
    @cache('_status')
    def _iterating_operator(self, ops, axis, merge_func=sum, init_val=0.):
        '''Execute a list of ops on X given the axis or axes'''
        if axis is not None:
            axis = _validate_operate_axis(axis)
        if not isinstance(ops, (tuple, list)):
            ops = [ops]

        # init values all zeros
        s = None
        # less than million data points, not a big deal
        for X in iter(self):
            if s is None:
                s = [o(X, axis) for o in ops]
            else:
                s = [merge_func((i, o(X, axis))) for i, o in zip(s, ops)]
        return s

    @cache('_status')
    def sum(self, axis=0):
        ops = lambda x, axis: np.sum(x, axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache('_status')
    def cumsum(self, axis=None):
        return self._dataset[:].cumsum(axis)

    @cache('_status')
    def sum2(self, axis=0):
        ops = lambda x, axis: np.sum(np.power(x, 2), axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache('_status')
    def pow(self, y):
        return self._dataset[:].__pow__(y)

    @cache('_status')
    def min(self, axis=None):
        ops = lambda x, axis: np.min(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] < x[1], x[0], x[1]),
            init_val=float('inf'))[0]

    @cache('_status')
    def argmin(self, axis=None):
        return self._dataset[:].argmin(axis)

    @cache('_status')
    def max(self, axis=None):
        ops = lambda x, axis: np.max(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] > x[1], x[0], x[1]),
            init_val=float('-inf'))[0]

    @cache('_status')
    def argmax(self, axis=None):
        return self._dataset[:].argmax(axis)

    @cache('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self.shape[i] for i in axis])
        return sum1 / n

    @cache('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._iterate_update(mean, 'sub')
        self._iterate_update(std, 'div')
        return self

    # ==================== Special operators ==================== #
    def _iterate_update(self, y, ops):
        # custom batch_size
        idx = list(range(0, self.shape[0], 1024))
        if idx[-1] < self.shape[0]:
            idx.append(self.shape[0])
        idx = list(zip(idx, idx[1:]))

        for i in idx:
            start, end = i
            if 'add' == ops:
                self._dataset[start:end] += y
            elif 'mul' == ops:
                self._dataset[start:end] *= y
            elif 'div' == ops:
                self._dataset[start:end] /= y
            elif 'sub' == ops:
                self._dataset[start:end] -= y
            elif 'floordiv' == ops:
                self._dataset[start:end] //= y
            elif 'pow' == ops:
                self._dataset[start:end] **= y

    def __add__(self, y):
        return self._mmap.__add__(y)

    def __sub__(self, y):
        return self._mmap.__sub__(y)

    def __mul__(self, y):
        return self._mmap.__mul__(y)

    def __div__(self, y):
        return self._mmap.__div__(y)

    def __floordiv__(self, y):
        return self._mmap.__floordiv__(y)

    def __pow__(self, y):
        return self._mmap.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._iterate_update(y, 'add')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._iterate_update(y, 'sub')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._iterate_update(y, 'mul')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._iterate_update(y, 'div')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._iterate_update(y, 'floordiv')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        self._iterate_update(y, 'pow')
        return self

    def __neg__(self):
        self._dataset.__neg__()
        return self

    def __pos__(self):
        self._dataset.__pos__()
        return self

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        return self._dataset.__getitem__(y)

    @autoattr(_status=lambda x: x + 1)
    def __setitem__(self, x, y):
        return self._dataset.__setitem__(x, y)

    # ==================== iteration ==================== #
    def __iter__(self):
        batch_size = self._batch_size
        seed = self._seed
        rng = self._rng

        # custom batch_size
        start = (int(self._start * self.shape[0]) if self._start < 1. + 1e-12
                 else int(self._start))
        end = (int(self._end * self.shape[0]) if self._end < 1. + 1e-12
               else int(self._end))
        if start > self.shape[0] or end > self.shape[0]:
            raise ValueError('start={} or end={} excess data_size={}'
                             ''.format(start, end, self.shape[0]))

        idx = list(range(start, end, batch_size))
        if idx[-1] < end:
            idx.append(end)
        idx = list(zip(idx, idx[1:]))
        if seed is not None:
            rng.seed(seed)
            rng.shuffle(idx)
            self._seed = None

        for i in idx:
            start, end = i
            yield self._dataset[start:end]

    # ==================== Strings ==================== #
    def __str__(self):
        return self._dataset.__str__()

    def __repr__(self):
        return self._dataset.__repr__()

    # ==================== Save ==================== #
    def resize(self, shape):
        if self._hdf.mode == 'r':
            return

        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], self.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], self.shape[1:]))
        if shape[0] < self.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self.shape[0]:
            return self

        self._dataset.resize(self.shape[0] + shape[0], 0)
        return self

    def flush(self):
        if self._hdf.mode == 'r':
            return
        self._hdf.flush()
