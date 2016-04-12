# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division
import os

import numpy as np
from scipy import stats

from ..io import Hdf5Data, MmapData, dataset, DataIterator
from .. import tensor

from itertools import izip
import h5py
import unittest
import time


# ======================================================================
# Run batch test first
# ======================================================================
def test_ops(data, assertFunc):
    a = data.array
    for i in [0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        for j in ['sum', 'cumsum', 'min', 'max', 'argmin', 'argmax', 'mean', 'var', 'std']:
            try:
                x = getattr(a, j)(axis=i).tolist()
                y = getattr(data, j)(axis=i).tolist()
                assertFunc(np.allclose(x, y), True, 'ops=' + j + ' axis=' + str(i))
            except Exception, e:
                pass
                # print('Exception:', e, i, j)


def cleanUp():
    if os.path.exists('tmp.h5'):
        os.remove('tmp.h5')
    if os.path.exists('tmp1.h5'):
        os.remove('tmp1.h5')
    if os.path.exists('tmp2.h5'):
        os.remove('tmp2.h5')

    if os.path.exists('test.h5'):
        os.remove('test.h5')
    if os.path.exists('test1.h5'):
        os.remove('test1.h5')
    if os.path.exists('test2.h5'):
        os.remove('test2.h5')

    if os.path.exists('tmp.ds'):
        os.remove('tmp.ds')

    if os.path.exists('test.ds'):
        os.remove('test.ds')
    if os.path.exists('test1.ds'):
        os.remove('test1.ds')
    if os.path.exists('test2.ds'):
        os.remove('test2.ds')
    if os.path.exists('test3.ds'):
        os.remove('test3.ds')

# ==================== Dataset ==================== #
path = 'set01'


def write():
    for f in os.listdir(path):
        try:
            os.remove(os.path.join(path, f))
        except:
            pass

    ds = dataset(path)
    # ====== set X ====== #
    x = ds.get_data('X1', dtype='float32', shape=(10000, 5), datatype='mmap')
    x[:] = np.arange(10000 * 5 * 0, 10000 * 5 * 1).reshape(-1, 5)
    x = ds.get_data('X2', dtype='float32', shape=(20000, 5), datatype='mmap')
    x[:] = np.arange(10000 * 5 * 1, 10000 * 5 * 3).reshape(-1, 5)
    x = ds.get_data('X3', dtype='float32', shape=(30000, 5), datatype='mmap')
    x[:] = np.arange(10000 * 5 * 3, 10000 * 5 * 6).reshape(-1, 5)

    # ====== set X ====== #
    y = ds.get_data('Y1', dtype='float32', shape=(10000, 5), datatype='mmap')
    y[:] = -np.arange(10000 * 5 * 0, 10000 * 5 * 1).reshape(-1, 5)
    y = ds.get_data('Y2', dtype='float32', shape=(20000, 5), datatype='mmap')
    y[:] = -np.arange(10000 * 5 * 1, 10000 * 5 * 3).reshape(-1, 5)
    y = ds.get_data('Y3', dtype='float32', shape=(30000, 5), datatype='mmap')
    y[:] = -np.arange(10000 * 5 * 3, 10000 * 5 * 6).reshape(-1, 5)

    # ====== set Z ====== #
    z = ds.get_data('Z', dtype='float32', shape=(60000, 5), datatype='mmap')
    z[:] = np.arange(0, 10000 * 5 * 6).reshape(-1, 5)

    ds.flush()
    # print("\nEstimated size:", ds.size)


def read(test):
    ds = dataset(path)
    # ====== read X ====== #
    x1 = ds.get_data('X1')
    x2 = ds.get_data('X2')
    x3 = ds.get_data('X3')
    test.assertEqual((x1.shape[0], x2.shape[0], x3.shape[0]),
                     (10000, 20000, 30000))
    # ====== read Y ====== #
    y1 = ds.get_data('Y1')
    y2 = ds.get_data('Y2')
    y3 = ds.get_data('Y3')
    test.assertEqual((y1.shape[0], y2.shape[0], y3.shape[0]),
                     (10000, 20000, 30000))
    # ====== read Z ====== #
    z = ds.get_data('Z')
    # ====== test iterator ====== #
    it1 = DataIterator((x1, x2, x3), seed=12, shuffle=True)
    it2 = DataIterator((y1, y2, y3), seed=12, shuffle=True)
    it3 = DataIterator(z, shuffle=False)
    # ====== mode1 ====== #
    if True:
        count = 0
        for i, j in zip(it1.set_mode(sequential=True, distribution=0.8),
                        it2.set_mode(sequential=True, distribution=0.8)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1) and count == 48000
        count = 0
        for i, j in zip(it1.set_mode(sequential=False, distribution=0.8),
                        it2.set_mode(sequential=False, distribution=0.8)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1) and count == 48000
    # ====== mode2 ====== #
    if True:
        count = 0
        for i, j in zip(it1.set_mode(sequential=True, distribution='up'),
                        it2.set_mode(sequential=True, distribution='up')):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1) and count == 90000
        count = 0
        for i, j in zip(it1.set_mode(sequential=False, distribution='down'),
                        it2.set_mode(sequential=False, distribution='down')):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1) and count == 30000
    # ====== mode3 ====== #
    if True:
        dist = np.random.rand(3).tolist()
        count = 0
        for i, j in zip(it1.set_mode(sequential=True, distribution=dist),
                        it2.set_mode(sequential=True, distribution=dist)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1), '{} != {}, dist={}'.format(count, len(it1), dist)
        count = 0
        for i, j in zip(it1.set_mode(sequential=False, distribution=dist),
                        it2.set_mode(sequential=False, distribution=dist)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1), '{} != {}, dist={}'.format(count, len(it1), dist)
    # ====== mode4 ====== #
    if True:
        dist = np.random.rand(3).tolist()
        dist[-1] = 0
        count = 0
        for i, j in zip(it1.set_mode(sequential=True, distribution=dist).set_range(0.2, 0.6),
                        it2.set_mode(sequential=True, distribution=dist).set_range(0.2, 0.6)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1), '{} != {}, dist={}'.format(count, len(it1), dist)
        count = 0
        for i, j in zip(it1.set_mode(sequential=False, distribution=dist).set_range(0.2, 0.6),
                        it2.set_mode(sequential=False, distribution=dist).set_range(0.2, 0.6)):
            assert i.shape == j.shape
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1), '{} != {}, dist={}'.format(count, len(it1), dist)
    # ====== mode5 ====== #
    if True: # len must diviable to batch_size
        count = 0
        for i, j in zip(it1.set_mode(sequential=True, distribution=1.).set_batch(200, shuffle=False).set_range(0., 1.),
                        it3.set_mode(sequential=True, distribution=1.).set_batch(200, shuffle=False)):
            assert i.shape == j.shape, '{} != {}'.format(i.shape, j.shape)
            assert np.sum(np.abs(i - j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1)
        count = 0
        for i, j in zip(it2.set_mode(sequential=True, distribution=1.).set_batch(200, shuffle=False).set_range(0., 1.),
                        it3.set_mode(sequential=True, distribution=1.).set_batch(200, shuffle=False)):
            assert i.shape == j.shape, '{} != {}'.format(i.shape, j.shape)
            assert np.sum(np.abs(i + j)) == 0., 'Difference:{}'.format(np.sum(np.abs(i + j)))
            count += i.shape[0]
        assert count == len(it1)


class DatasetTest(unittest.TestCase):

    def tearDown(self):
        cleanUp()

    def test_mmap_data(self):
        data = MmapData('test', shape=(50, 5, 5), mode='w+', dtype='float64')
        for i in range(data.shape[0]):
            data[i] = i

        x = data.sum(0)
        y = data.sum(0)
        self.assertEqual(x.tolist(), y.tolist())

        x = data.mean(0)
        y = data.mean(0)
        data += 1; z = data.mean(0)
        self.assertEqual(x.tolist(), y.tolist())
        self.assertNotEqual(x.tolist(), z.tolist())
        test_ops(data, self.assertEqual)

        # test normalization
        data.normalize((0, 1))
        self.assertEqual(np.allclose(data.mean((0, 1)), 0.), True)
        self.assertEqual(np.allclose(data.std((0, 1)), 1.), True)

        # test iteration
        data.set_batch(8, 12)
        count = 0
        for x in data:
            count += x.shape[0]
        self.assertEqual(count, data.shape[0])

        # test prepend
        x = np.arange(0, 250).reshape(-1, 5, 5)
        data.prepend(x)
        self.assertEqual(x.tolist(), data[:x.shape[0]].tolist())

        x = np.arange(0, 25 * (data.shape[0] + 12)).reshape(-1, 5, 5)
        data.prepend(x)
        data.flush()
        data = MmapData(data.path)
        self.assertEqual(x.tolist(), data[:x.shape[0]].tolist())

        # end test
        os.remove(data.path)

    def test_hdf5_data(self):
        f = h5py.File('test.h5', mode='w')
        data = Hdf5Data('X', f, shape=(None, 5, 5), dtype='float64')
        data.append(np.random.rand(500, 5, 5), np.random.rand(200, 5, 5),
            np.random.rand(10, 5, 8))
        x = data.sum(0)
        y = data.sum(0)
        self.assertEqual(x.tolist(), y.tolist())

        x = data.mean(0)
        y = data.mean(0)
        data += 1; z = data.mean(0)
        self.assertEqual(x.tolist(), y.tolist())
        self.assertNotEqual(x.tolist(), z.tolist())
        test_ops(data, self.assertEqual)

        # test normalization
        data.normalize((0, 1))
        self.assertEqual(np.allclose(data.mean((0, 1)), 0.), True)
        self.assertEqual(np.allclose(data.std((0, 1)), 1.), True)

        # test iteration
        data.set_batch(8, 12)
        count = 0
        for x in data:
            count += x.shape[0]
        self.assertEqual(count, data.shape[0])

        # test prepend
        x = np.arange(0, 250).reshape(-1, 5, 5)
        data.prepend(x)
        data.flush()
        self.assertEqual(x.tolist(), data[:x.shape[0]].tolist())

        x = np.arange(0, 25 * (data.shape[0] + 12)).reshape(-1, 5, 5)
        data.prepend(x)
        data.flush()
        self.assertEqual(x.tolist(), data[:x.shape[0]].tolist())

    def test_dataset(self):
        write()
        for i in range(10):
            read(self)
        try:
            os.remove(path)
        except:
            pass


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
