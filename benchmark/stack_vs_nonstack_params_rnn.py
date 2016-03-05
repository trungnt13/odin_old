# ===========================================================================
# Conclusion:
# Stack parameters and precompute_inputs significantly increase speed
# ===========================================================================
from __future__ import print_function, division

import os
os.environ['ODIN'] = 'cpu,float32,theano'
import odin
from odin import tensor as T
import numpy as np
import time

batch_size = 128
seq_len = 512
X = T.variable(np.random.rand(batch_size, seq_len, 20))

W1 = T.variable(np.random.rand(20, 10))
W2 = T.variable(np.random.rand(20, 10))
W3 = T.variable(np.random.rand(20, 10))
W4 = T.variable(np.random.rand(20, 10))

hidden = T.variable(np.random.rand(batch_size, 20))
# ====== First approach ====== #
W = T.concatenate((W1, W2, W3, W4), axis=1) # 20x40
inputs = T.dot(X, W) #batch_sizexseq_lenx40

inputs1 = T.dot(X, W1)
inputs2 = T.dot(X, W2)
inputs3 = T.dot(X, W3)
inputs4 = T.dot(X, W4)


def step1(x, h, mem):
    x1 = x[:, 0:10] # seq_lenx10
    x2 = x[:, 10:20]
    x3 = x[:, 20:30]
    x4 = x[:, 30:40]

    h = T.dot(h.dimshuffle('x', 0), W)
    h1 = h[:, 0:10] # 1x10
    h2 = h[:, 10:20]
    h3 = h[:, 20:30]
    h4 = h[:, 30:40]

    test = (T.dot(x1, T.transpose(h1)) +
            T.dot(x2, T.transpose(h2)) +
            T.dot(x3, T.transpose(h3)) +
            T.dot(x4, T.transpose(h4)))
    return test + mem

mem, updates = T.scan(step1,
                sequences=[inputs, hidden],
                outputs_info=[T.zeros(shape=(seq_len, 1))]
            )
f1 = T.function(inputs=[], outputs=mem)
print(f1().shape)

# ====== second approach ====== #


def step2(x1, x2, x3, x4, h, mem):
    h = h.dimshuffle('x', 0) # 1x20
    h1 = T.dot(h, W1) # 1x10
    h2 = T.dot(h, W2)
    h3 = T.dot(h, W3)
    h4 = T.dot(h, W4)
    test = (T.dot(x1, T.transpose(h1)) +
            T.dot(x2, T.transpose(h2)) +
            T.dot(x3, T.transpose(h3)) +
            T.dot(x4, T.transpose(h4)))
    return test + mem
mem, updates = T.scan(step2,
                sequences=[inputs1, inputs2, inputs3, inputs4, hidden],
                outputs_info=[T.zeros(shape=(seq_len, 1))]
            )
f2 = T.function(inputs=[], outputs=mem)
print(f2().shape)

# ====== third approach ====== #


def step3(x, h, mem):
    x1 = x[:, 0:10] # seq_lenx10
    x2 = x[:, 10:20]
    x3 = x[:, 20:30]
    x4 = x[:, 30:40]

    h = h.dimshuffle('x', 0) # 1x20
    h1 = T.dot(h, W1) # 1x10
    h2 = T.dot(h, W2)
    h3 = T.dot(h, W3)
    h4 = T.dot(h, W4)
    test = (T.dot(x1, T.transpose(h1)) +
            T.dot(x2, T.transpose(h2)) +
            T.dot(x3, T.transpose(h3)) +
            T.dot(x4, T.transpose(h4)))
    return test + mem
mem, updates = T.scan(step3,
                sequences=[inputs, hidden],
                outputs_info=[T.zeros(shape=(seq_len, 1))]
            )
f3 = T.function(inputs=[], outputs=mem)
print(f3().shape)


# ====== benchmark ====== #
start = time.time()
for i in xrange(20):
    f1()
print('Time:', time.time() - start)

start = time.time()
for i in xrange(20):
    f2()
print('Time:', time.time() - start)

start = time.time()
for i in xrange(20):
    f3()
print('Time:', time.time() - start)

# Time: 0.377168178558
# Time: 0.464277029037
# Time: 0.40033698082
