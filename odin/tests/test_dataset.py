# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division
import os

import numpy as np
from scipy import stats

from ..io import Hdf5Data, MmapData, dataset
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


class BatchTest(unittest.TestCase):

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


class DatasetTest(unittest.TestCase):

    def tearDown(self):
        cleanUp()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
