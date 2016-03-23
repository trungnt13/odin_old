# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from .. import tensor as T
from ..objectives import *
from ..optimizers import *
from ..metrics import *

# ===========================================================================
# Main Test
# ===========================================================================


class ObjectivesTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_all_objectives(self):
        np.random.seed(12082518)
        y_pred = np.random.rand(16, 10)
        y_true = np.random.rand(16, 10)
        y_pred_ = T.variable(y_pred)
        y_true_ = T.variable(y_true)

        def calc_cost(func):
            cost = T.mean(func(y_pred_, y_true_))
            diffable = False
            try:
                grad = T.gradients(cost, y_pred_)[0]
                grad = T.eval(T.mean(grad))
                if grad == 0.:
                    print('%-24s' % func.__name__, "Non-differentiable!")
                else:
                    print('%-24s' % func.__name__, "Differentiable!")
                    diffable = True
            except:
                print('%-24s' % func.__name__, "Non-Differentiable!")
            return round(T.eval(cost).tolist(), 3), diffable
        print()
        self.assertEqual(calc_cost(squared_loss), (0.174, True))
        self.assertEqual(calc_cost(absolute_loss), (0.351, True))
        self.assertEqual(calc_cost(absolute_percentage_loss), (305.868, True))
        self.assertEqual(calc_cost(squared_logarithmic_loss), (0.080, True))
        self.assertEqual(calc_cost(squared_hinge), (0.567, True))
        self.assertEqual(calc_cost(hinge), (0.715, True))
        self.assertEqual(calc_cost(categorical_crossentropy), (13.930, True))
        self.assertEqual(calc_cost(poisson), (1.035, True))
        self.assertEqual(calc_cost(cosine_proximity), (-0.078, True))
        self.assertEqual(calc_cost(hinge), (0.715, True))

        self.assertEqual(calc_cost(binary_accuracy), (0.438, False))
        self.assertEqual(calc_cost(categorical_accuracy), (0.063, False))


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_updates_constraint(self):
        # ====== simple matrix factorization ====== #
        np.random.seed(12082518)
        x = np.random.rand(16, 8)
        y = np.random.rand(8, 13)
        z = np.random.rand(16, 13)

        x_ = T.variable(x)
        y_ = T.variable(y)
        z_pred = T.dot(x_, y_)
        z_true = T.variable(z)
        obj = T.mean(squared_loss(z_pred, z_true))
        params = [x_, y_]

        def test_opt(updates):
            T.set_value(x_, x)
            T.set_value(y_, y)
            f = T.function(
                inputs=[],
                outputs=obj,
                updates=updates)
            r = [f() for i in xrange(30)]
            self.assertGreater(r[:-1], r[1:])
        print()
        print('%-18s OK!' % 'sgd'); test_opt(sgd(obj, params, 0.1))
        print('%-18s OK!' % 'momentum'); test_opt(momentum(obj, params, 0.1, 0.9))
        print('%-18s OK!' % 'nesterov_momentum'); test_opt(nesterov_momentum(obj, params, 0.1, 0.9))
        print('%-18s OK!' % 'adagrad'); test_opt(adagrad(obj, params, 0.1, 1e-6))
        print('%-18s OK!' % 'adadelta'); test_opt(adadelta(obj, params, 0.1, 0.95, 1e-6))
        print('%-18s OK!' % 'rmsprop'); test_opt(rmsprop(obj, params, 0.1, 0.9, 1e-6))
        print('%-18s OK!' % 'adamax'); test_opt(adamax(obj, params, 0.1, 0.9, 0.999, 1e-8))
        print('%-18s OK!' % 'adam'); test_opt(adam(obj, params, 0.1, 0.9, 0.999, 1e-8))

        g = T.gradients(obj, params)
        total_norm_constraint(g, 2.)
        print('%-18s OK!' % 'sgd+norm lr=0.1'); test_opt(sgd(g, params, 0.1))
        print('%-18s OK!' % 'sgd+norm lr=1.'); test_opt(sgd(g, params, 1.))

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
