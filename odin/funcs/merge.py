from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction

class Summation(OdinFunction):

    def __init__(self, incoming):
        super(Summation, self).__init__(incoming)

    @property
    def output_shape(self):
        return self.input_shape[0]

    def __call__(self, training=False):
        X = self.get_inputs(training=training)
        return sum(x for x in X)

    def get_cost(self, objectives=None, unsupervised=False, training=True):
        if objectives is None or not hasattr(objectives, '__call__'):
            raise ValueError('objectives must be a function!')

        y_pred = self(training)
        if unsupervised:
            output_var = self.input_var
        else:
            output_var = self.output_var
        return objectives(y_pred, *output_var)
