from __future__ import print_function, division
import numpy as np
import os
os.environ['ODIN'] = 'theano,float32'
from odin import tensor as T
import time
import theano
def step(s1, s2, s3, o1, o2, n1, n2):
    return o1, o2

seq1 = T.variable(np.arange(10))
seq2 = T.variable(np.arange(20))
seq3 = T.variable(np.arange(5))

nonseq1 = T.variable(1.)
nonseq2 = T.variable(2.)

([o1, o2], updates) = theano.scan(
    fn=step,
    sequences=[seq1, seq2, seq3],
    outputs_info=[T.zeros((2, 2)), T.ones((2, 2))],
    non_sequences=[nonseq1, nonseq2],
    n_steps=None,
    truncate_gradient=-1,
    go_backwards=False)

f1 = T.function(
    inputs=[],
    outputs=[o1, o2],
    updates=updates)
a, b = f1()
print(a.shape)

o1, o2 = T.loop(step,
    sequences=[seq1, seq2, seq3],
    outputs_info=[T.zeros((2, 2)), T.ones((2, 2))],
    non_sequences=[nonseq1, nonseq2],
    n_steps=5)
print(o1, o2)
f2 = T.function(
    inputs=[],
    outputs=[o1, o2],
    updates=updates)
a, b = f2()
print(a.shape)

# ====== performance ====== #
t = time.time()
for i in xrange(10):
    f1()
print('Time:', time.time() - t)

t = time.time()
for i in xrange(10):
    f2()
print('Time:', time.time() - t)

# Time: 0.00640487670898
# Time: 0.000237941741943