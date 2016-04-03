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
from .. import config

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
        y_pred = np.random.rand(16, 10).astype(config.floatX())
        y_true = np.random.rand(16, 10).astype(config.floatX())
        y_pred_ = T.variable(y_pred)
        y_true_ = T.variable(y_true)

        round_value = 3
        if config.floatX() == 'float16':
            y_pred_ = T.cast(y_pred_, 'float32')
            y_true_ = T.cast(y_true_, 'float32')
            round_value = 1

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
            return round(T.eval(cost).tolist(), round_value), diffable

        print()
        self.assertEqual(calc_cost(squared_loss),
                        (round(0.174, round_value), True))
        self.assertEqual(calc_cost(absolute_loss),
                        (round(0.351, round_value), True))
        self.assertEqual(calc_cost(absolute_percentage_loss),
                        (round(305.868, round_value), True))
        self.assertEqual(calc_cost(squared_logarithmic_loss),
                        (round(0.080, round_value), True))
        self.assertEqual(calc_cost(squared_hinge),
                        (round(0.567, round_value), True))
        self.assertEqual(calc_cost(hinge),
                        (round(0.715, round_value), True))
        self.assertEqual(calc_cost(categorical_crossentropy),
                        (round(13.930, round_value), True))
        self.assertEqual(calc_cost(poisson),
                        (round(1.035, round_value), True))
        self.assertEqual(calc_cost(cosine_proximity),
                        (round(-0.078, round_value), True))
        self.assertEqual(calc_cost(hinge),
                        (round(0.715, round_value), True))
        self.assertEqual(calc_cost(bayes_crossentropy),
                        (round(5.008, round_value), True))
        self.assertEqual(calc_cost(hellinger_distance),
                        (round(0.734, round_value), True))

        self.assertEqual(calc_cost(binary_accuracy),
                        (round(0.438, round_value), False))
        self.assertEqual(calc_cost(categorical_accuracy),
                        (round(0.063, round_value), False))


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_updates_constraint(self):
        # ====== simple matrix factorization ====== #
        np.random.seed(12082518)
        x = np.round(np.random.rand(16, 8), 5).astype(config.floatX())
        y = np.round(np.random.rand(8, 13), 5).astype(config.floatX())
        z = np.round(np.random.rand(16, 13), 5).astype(config.floatX())

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
            r = [i if isinstance(i, float) else i.tolist() for i in r]
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
