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

        num_inputs = self._validate_nD_input(2)
        shape = (num_inputs[1], num_units)

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
        return [(i[0], self.num_units) for i in self.input_shape]

    def __call__(self, training=False):
        inputs = self.get_inputs(training)
        outputs = []
        # ====== processing each inputs ====== #
        for x in inputs:
            if T.ndim(x) > 2:
                # if the input has more than two dimensions, flatten it into a
                # batch of feature vectors.
                x = T.flatten(x, 2)
            activation = T.dot(x, self.W())
            if self.b is not None:
                activation = activation + T.reshape(self.b(), (1, -1))
            outputs.append(self.nonlinearity(activation))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
