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
        pass

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
