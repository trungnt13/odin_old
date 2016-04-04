# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
os.environ['THEANO_FLAGS'] = \
'contexts=dev0->cuda0;dev1->cuda1,' + \
'device=gpu,' + \
'mode=FAST_RUN,floatX=float32,exception_verbosity=high'
# os.environ['THEANO_FLAGS'] = \
# 'device=gpu,mode=FAST_RUN,floatX=float32,exception_verbosity=high'

import numpy as np
import theano
from theano import tensor as T
import time

np.random.seed(12082518)
x = np.random.rand(1000, 10)
x_gpuarray0 = theano.shared(x.astype('float32'), borrow=False, target='dev0')
x_gpuarray1 = theano.shared(x.astype('float32'), borrow=False, target='dev1')
x_cuda = theano.shared(x.astype('float32'), borrow=False)
x_tensor = theano.shared(x.astype('float32'), target='cpu')
print(x_gpuarray0, x_gpuarray0.__class__)
print(x_gpuarray1, x_gpuarray1.__class__)
print(x_cuda, x_tensor.__class__)
print(x_tensor, x_tensor.__class__)


def step(x, o):
    return x**2 + o


out = T.zeros((10,))
output, updates = theano.scan(step, sequences=x_gpuarray0, outputs_info=out)
f_gpuarray0 = theano.function(inputs=[], outputs=output)

out = T.zeros((10,))
output, updates = theano.scan(step, sequences=x_gpuarray1, outputs_info=out)
f_gpuarray1 = theano.function(inputs=[], outputs=output)

out = T.zeros((10,))
output, updates = theano.scan(step, sequences=x_cuda, outputs_info=out)
f_cuda = theano.function(inputs=[], outputs=output)

out = T.zeros((10,))
output, updates = theano.scan(step, sequences=x_tensor, outputs_info=out)
f_tensor = theano.function(inputs=[], outputs=output)


# ====== benchmark ====== #
print()
t = time.time()
for i in range(10):
    x0 = f_gpuarray0()
print(x_gpuarray0.__class__.__name__, ':', time.time() - t)

t = time.time()
for i in range(10):
    x1 = f_gpuarray1()
print(x_gpuarray1.__class__.__name__, ':', time.time() - t)

t = time.time()
for i in range(10):
    x2 = f_cuda()
print(x_cuda.__class__.__name__, ':', time.time() - t)

t = time.time()
for i in range(10):
    x3 = f_tensor()
print(x_tensor.__class__.__name__, ':', time.time() - t)

print('Identical results:',
     ((np.sum(np.abs(x1 - x2)) == 0.) and
      (np.sum(np.abs(x1 - x3)) == 0.) and
      (np.sum(np.abs(x1 - x0)) == 0.)))
