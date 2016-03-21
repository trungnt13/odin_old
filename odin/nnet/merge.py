from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction


class Summation(OdinFunction):

    def __init__(self, incoming, **kwargs):
        super(Summation, self).__init__(incoming, **kwargs)

    @property
    def output_shape(self):
        return [self.input_shape[0]]

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = [sum(x for x in inputs)]
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
