# ===========================================================================
# Benchmark results (200MB of data)
# Memmap create: 8.46418905258
# Hdf5 create: 11.3049161434
# Memmap traverse: 0.000792026519775
# Hdf5 traverse: 0.337000131607
# ===========================================================================

from __future__ import print_function, division
import os
import numpy as np
import h5py
import time

path = '.'
mm_path = os.path.join(path, 'tmp.mm')
h5_path = os.path.join(path, 'tmp.h5')

x = np.ones((100000, 500))
start = time.time()
x1 = np.memmap(mm_path, dtype='float32', mode='w+',
    shape=(100000, 500))
x1[:] = x
x1.flush()
print('Memmap create:', time.time() - start)

x2 = h5py.File(h5_path, mode='w')
x2.create_dataset('X', dtype='float32', chunks=(1000, 500), data=x)
x2.flush()
print('Hdf5 create:', time.time() - start)


idx = range(0, x.shape[0], 256)
idx = zip(idx, idx[1:])
np.random.shuffle(idx)

t = time.time()
for i in idx:
    start, end = i
    x1[start:end]
print('Memmap traverse:', time.time() - t)

t = time.time()
for i in idx:
    start, end = i
    x2['X'][start:end]
print('Hdf5 traverse:', time.time() - t)

# ====== remove everything ====== #
try:
    os.remove(mm_path)
    os.remove(h5_path)
except:
    pass
