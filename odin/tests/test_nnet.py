# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import nnet
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

class FunctionsTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)

    def tearDown(self):
        logger.set_enable(True)

    def test_dense_func(self):
        d3 = nnet.Dense((None, 10), num_units=5, nonlinearity=T.linear)
        f_pred = T.function(d3.input_var, d3())

        x = np.random.rand(16, 10)
        y = np.random.rand(16, 5)

        p = d3.get_params_value(True)[0]
        pred1 = np.round(f_pred(x)[0], 6)
        pred2 = np.round(np.dot(x, p), 6)
        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), 10e-5)

        # ====== only cost ====== #
        cost, _ = d3.get_optimization(
            objective=objectives.squared_loss,
            training=False)
        f_cost = T.function(d3.input_var + d3.output_var, cost)

        cost1 = np.round(f_cost(x, y), 6)
        cost2 = np.round((np.dot(x, p) - y)**2, 6)
        self.assertLessEqual(np.sum(np.abs(cost1 - cost2)), 10e-5)

        # ====== optimization ====== #
        cost, updates = d3.get_optimization(
            objective=objectives.mean_squared_loss,
            optimizer=optimizers.sgd)
        f_updates = T.function(
            inputs=d3.input_var + d3.output_var,
            outputs=cost,
            updates=updates)
        cost = []
        for i in range(10):
            cost.append(f_updates(x, y))
        self.assertGreater(cost[:-1], cost[1:])

    def test_summation_merge(self):
        d1 = nnet.Dense((None, 10), num_units=5, nonlinearity=T.linear)
        d2 = nnet.Dense((None, 20), num_units=5, nonlinearity=T.linear)
        d3 = nnet.Summation((d1, d2))

        params = d3.get_params_value(True)
        p1 = params[0]
        p2 = params[2]

        f_pred = T.function(d3.input_var, d3())

        x1 = np.random.rand(16, 10)
        x2 = np.random.rand(16, 20)

        pred1 = np.round(f_pred(x1, x2), 6)
        pred2 = np.round(np.dot(x1, p1) + np.dot(x2, p2), 6)

        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), 10e-5)

    def test_get_roots_and_children(self):
        d1a = nnet.Dense((None, 28, 28), num_units=256, name='d1a')
        d1b = nnet.Dense(d1a, num_units=128, name='d1b')
        d1c = nnet.Dense(d1b, num_units=128, name='d1c')
        d1d = nnet.Summation([(None, 128), d1c], name='Summation')

        self.assertEqual(d1d.incoming, [None, d1c])
        self.assertEqual(d1d.input_shape, [(None, 128), (None, 128)])
        self.assertEqual([T.ndim(i) for i in d1d.input_var], [2, 3])
        self.assertEqual(d1d.get_roots(), [d1d, d1a])
        self.assertEqual(d1d.get_children(), [d1c, d1b, d1a])

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
