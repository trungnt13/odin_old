# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from .. import funcs
from .. import tensor as T
from .. import logger
from .. import objectives

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
        d3 = funcs.Dense((None, 10), num_units=5)
        f_pred = T.function(d3.input_var, d3())

        x = np.random.rand(16, 10)
        y = np.random.rand(16, 5)

        p = d3.get_params_value(True)[0]
        pred1 = np.round(f_pred(x), 6)
        pred2 = np.round(np.dot(x, p), 6)
        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), 10e-5)

        cost = d3.get_cost(
            objectives=objectives.squared_loss,
            unsupervised=False,
            training=True)
        f_cost = T.function(d3.input_var + d3.output_var, cost)

        cost1 = np.round(f_cost(x, y), 6)
        cost2 = np.round((np.dot(x, p) - y)**2, 6)
        self.assertLessEqual(np.sum(np.abs(cost1 - cost2)), 10e-5)

    def test_summation_merge(self):
        d1 = funcs.Dense((None, 10), num_units=5)
        d2 = funcs.Dense((None, 20), num_units=5)
        d3 = funcs.Summation((d1, d2))

        params = d3.get_params_value(True)
        p1 = params[0]
        p2 = params[2]

        f_pred = T.function(d3.input_var, d3())

        x1 = np.random.rand(16, 10)
        x2 = np.random.rand(16, 20)

        pred1 = np.round(f_pred(x1, x2), 6)
        pred2 = np.round(np.dot(x1, p1) + np.dot(x2, p2), 6)

        self.assertLessEqual(np.sum(np.abs(pred1 - pred2)), 10e-5)

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
