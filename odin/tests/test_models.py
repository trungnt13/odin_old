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
        logger.set_enable(False)

        m = model('tmp.ai')
        m.set_model(_test, dim=10)
        ai1 = m.get_model()
        w1 = lasagne.layers.get_all_param_values(ai1)
        y1 = m.pred(X)
        m.save()

        m = model.load('tmp.ai')
        ai2 = m.get_model()
        w2 = lasagne.layers.get_all_param_values(ai2)
        y2 = m.pred(X)

        for i, j in zip(w1, w2):
            self.assertEqual(i.tolist(), j.tolist())
        self.assertEqual(y1[0].tolist(), y2[0].tolist())

        os.remove('tmp.ai')
        logger.set_enable(True)

    def test_save_load_keras(self):
        X = np.random.rand(32, 10)

        m = model('tmp.ai')
        m.set_model(_test_keras, dim=10)
        y1 = m.pred(X)
        m.save()

        m = model.load('tmp.ai')
        y2 = m.pred(X)

        self.assertEqual(y1[0].tolist(), y2[0].tolist())

        os.remove('tmp.ai')

    def test_convert_weights_lasagne_keras(self):
        X = np.random.rand(64, 10)

        m1 = model()
        m1.set_model(_test, dim=10)

        m2 = model()
        m2.set_model(_test_keras, dim=10)

        m2.set_weights(m1.get_params(), 'lasagne')
        self.assertEqual(m1.pred(X)[0].tolist(), m2.pred(X)[0].tolist())

    def test_convert_weights_keras_lasagne(self):
        X = np.random.rand(64, 10)

        m1 = model()
        m1.set_model(_test, dim=10)

        m2 = model()
        m2.set_model(_test_keras, dim=10)

        m1.set_weights(m2.get_params(), 'keras')
        self.assertEqual(np.round(m1.pred(X), 6)[0].tolist(),
                         np.round(m2.pred(X), 6)[0].tolist())

# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    print(' Use nosetests to run these tests ')
