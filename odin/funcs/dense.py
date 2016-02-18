from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction

class Dense(OdinFunction):

    def __init__(self, incoming, num_units,
                 W=T.random_uniform, b=T.zeros, nonlinearity=T.relu):
        super(Dense, self).__init__(incoming)
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

    def get_cost(self, objectives=None, unsupervised=False, training=True):
        if objectives is None or not hasattr(objectives, '__call__'):
            raise ValueError('objectives must be a function!')
        y_pred = self(training)
        if unsupervised:
            output_var = self.input_var
        else:
            output_var = self.output_var
        return objectives(y_pred, *output_var)

    def __call__(self, training=False):
        X = self.get_inputs(training=True)[0]
        if T.ndim(X) > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            X = T.flatten(X, 2)

        activation = T.dot(X, self.W)
        if self.b is not None:
            activation = activation + T.reshape(self.b, (1, -1))
        return self.nonlinearity(activation)
