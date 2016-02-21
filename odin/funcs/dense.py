from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction

class Dense(OdinFunction):

    def __init__(self, incoming, num_units,
                 W=T.np_symmetric_uniform,
                 b=T.np_constant,
                 nonlinearity=T.relu,
                 unsupervised=False,
                 **kwargs):
        super(Dense, self).__init__(
            incoming, unsupervised=unsupervised, **kwargs)

        shape = (np.prod(self.input_shape[0][1:]), num_units)
        self.W = self.create_params(
            W, shape, 'W', regularizable=True, trainable=True)
        if b is None:
            self.b = None
        else:
            self.b = self.create_params(
                b, (num_units,), 'b', regularizable=False, trainable=True)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

    @property
    def output_shape(self):
        return (self.input_shape[0][0], self.num_units)

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        return self._deterministic_optimization_procedure(
            objective, optimizer, globals, training)

    def __call__(self, training=False, **kwargs):
        X = self.get_inputs(training=True)
        activation = T.castX(0.)
        for x in X:
            if T.ndim(x) > 2:
                # if the input has more than two dimensions, flatten it into a
                # batch of feature vectors.
                x = T.flatten(x, 2)

            activation = activation + T.dot(x, self.W)
            if self.b is not None:
                activation = activation + T.reshape(self.b, (1, -1))
        return self.nonlinearity(activation)
