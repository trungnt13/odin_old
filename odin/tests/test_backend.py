# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import funcs
from .. import tensor as T
from .. import logger
from .. import objectives
from .. import optimizers

import unittest

import os
import numpy as np

from six.moves import zip, range

# ===========================================================================
# Main Test
# ===========================================================================

class BackendTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)

    def tearDown(self):
        logger.set_enable(True)

    def test_set_subtensor(self):
        x = T.variable(np.zeros((10, 10)))
        y = T.variable(np.ones((10, 10)))
        z = T.eval(T.set_subtensor(x[:, :], y[:, :]))

        self.assertEqual(z.ravel().tolist(), [1.] * 100)

    def test_gradients(self):
        a = T.variable(1.2)
        b = T.variable(1.3)
        x = a * b

        y = T.variable(2.)
        z = a + b

        true = T.variable(3.)
        pred = x * y * z
        loss = T.pow((true - pred), 2)

        G = T.gradients(loss, [a, b, y],
            consider_constant=[x], known_grads={a: T.ones_like(a)})
        G = [T.eval(g) for g in G]
        G = [g.tolist() if isinstance(g, np.ndarray) else g for g in G]
        G = [round(g, 6) for g in G]
        self.assertEqual(G, [1.0, 29.951998, 37.439999])

    def test_loop(self):
        pass

    def test_scan(self):
        def step(s1, s2, s3, o1, o2, n1, n2):
            return o1, o2

        seq1 = T.variable(np.arange(10))
        seq2 = T.variable(np.arange(20))
        seq3 = T.variable(np.arange(5))

        nonseq1 = T.variable(1.)
        nonseq2 = T.variable(2.)

        ([o1, o2], updates) = T.scan(step,
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

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
