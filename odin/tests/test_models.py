# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

from ..utils import function, frame
from ..model import model
from .. import tensor
from .. import logger
import unittest

import os
import numpy as np

from six.moves import zip, range

# ===========================================================================
# Main Test
# ===========================================================================


def _test(dim):
    import lasagne
    l = lasagne.layers.InputLayer(shape=(None, dim))
    l = lasagne.layers.DenseLayer(l, num_units=64)
    l = lasagne.layers.DenseLayer(l, num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax)
    return l


def _test_keras(dim):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    model = Sequential()
    model.add(Dense(64, input_shape=(dim,)))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model


class ModelTest(unittest.TestCase):

    def setUp(self):
        logger.set_enable(False)

    def tearDown(self):
        logger.set_enable(True)
        if os.path.exists('tmp.ai'):
            os.remove('tmp.ai')

    def test_save_load_lasagne(self):
        import lasagne
        X = np.random.rand(32, 10)
        y = np.random.rand(32, 3)
        logger.set_enable(False)

        m = model('tmp.ai')
        m.set_model(_test, dim=10)
        m.create_cost(tensor.categorical_crossentropy)
        ai1 = m.get_model()
        w1 = lasagne.layers.get_all_param_values(ai1)
        y1 = m.pred(X)
        c1 = m.cost(X, y)
        m.save()

        m = model.load('tmp.ai')
        m.create_cost(tensor.categorical_crossentropy)
        ai2 = m.get_model()
        w2 = lasagne.layers.get_all_param_values(ai2)
        y2 = m.pred(X)
        c2 = m.cost(X, y)

        for i, j in zip(w1, w2):
            self.assertEqual(i.tolist(), j.tolist())
        self.assertEqual(y1.tolist(), y2.tolist())
        self.assertEqual(c1.tolist(), c2.tolist())

        os.remove('tmp.ai')
        logger.set_enable(True)

    def test_save_load_keras(self):
        X = np.random.rand(32, 10)
        y = np.random.rand(32, 3)

        m = model('tmp.ai')
        m.set_model(_test_keras, dim=10)
        m.create_cost(tensor.categorical_crossentropy)
        y1 = m.pred(X)
        c1 = m.cost(X, y)
        m.save()

        m = model.load('tmp.ai')
        m.create_cost(tensor.categorical_crossentropy)
        y2 = m.pred(X)
        c2 = m.cost(X, y)

        self.assertEqual(y1.tolist(), y2.tolist())
        self.assertEqual(c1.tolist(), c2.tolist())

        os.remove('tmp.ai')

    def test_convert_weights_lasagne_keras(self):
        X = np.random.rand(64, 10)
        y = np.random.rand(64, 3)

        m1 = model()
        m1.set_model(_test, dim=10)
        m1.create_cost(tensor.categorical_crossentropy)

        m2 = model()
        m2.set_model(_test_keras, dim=10)
        m2.create_cost(tensor.categorical_crossentropy)

        m2.set_weights(m1.get_params(), 'lasagne')
        self.assertEqual(m1.pred(X).tolist(), m2.pred(X).tolist())
        self.assertEqual(m1.cost(X, y).tolist(),
                         m2.cost(X, y).tolist())

    def test_convert_weights_keras_lasagne(self):
        X = np.random.rand(64, 10)
        y = np.random.rand(64, 3)

        m1 = model()
        m1.set_model(_test, dim=10)
        m1.create_cost(tensor.categorical_crossentropy)

        m2 = model()
        m2.set_model(_test_keras, dim=10)
        m2.create_cost(tensor.categorical_crossentropy)

        m1.set_weights(m2.get_params(), 'keras')
        self.assertEqual(np.round(m1.pred(X), 6).tolist(),
                         np.round(m2.pred(X), 6).tolist())
        self.assertEqual(m1.cost(X, y).tolist(),
                         m2.cost(X, y).tolist())

    def test_updates_function(self):
        X = np.random.rand(64, 10)
        y = np.random.rand(64, 3)

        m1 = model()
        m1.set_model(_test, dim=10)
        m1.create_cost(tensor.categorical_crossentropy)
        w1 = m1.get_params()

        m2 = model()
        m2.set_model(_test_keras, dim=10)
        m2.create_cost(tensor.categorical_crossentropy)
        m2.set_weights(w1, 'lasagne')

        import lasagne
        m1.create_updates(tensor.categorical_crossentropy,
                          lasagne.updates.rmsprop)
        m2.create_updates(tensor.categorical_crossentropy,
                          lasagne.updates.rmsprop)
        m1.updates(X, y)
        m2.updates(X, y)
        self.assertEqual(m1.cost(X, y).tolist(), m2.cost(X, y).tolist())
# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
