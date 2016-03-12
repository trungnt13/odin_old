# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# This module is created based on the code from keras library
# Original work Copyright (c) 2014-2015 keras contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

import numpy as np

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple

__all__ = [
    'UpSampling1D',
    'UpSampling2D',
    'GlobalPool'
]


class UpSampling1D(OdinFunction):

    """
    1D upscaling layer

    Performs 1D upscaling over the trailing axis of a 3D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor. If an iterable, it should have one element.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor=2, axis=-1, **kwargs):
        super(UpSampling1D, self).__init__(incoming, unsupervised=False, **kwargs)
        self.scale_factor = scale_factor
        self.axis = axis

    @property
    def output_shape(self):
        outshape = []
        scale = self.scale_factor
        for shape in self.input_shape:
            axis = self.axis
            if self.axis < 0:
                axis = len(shape) - 1
            outshape.append(
                tuple(j * scale if i == axis and j is not None else j
                      for i, j in enumerate(shape)))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for X in inputs:
            outputs.append(
                T.repeat_elements(X, self.scale_factor, axis=self.axis))
        self._log_footprint(training, inputs, outputs)
        return outputs


class UpSampling2D(OdinFunction):

    """ 2D upscaling layer

    Performs 2D upscaling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(UpSampling2D, self).__init__(
            incoming, unsupervised=False, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    @property
    def output_shape(self):
        outshape = []
        for shape in self.input_shape:
            output_shape = list(shape)  # copy / convert to mutable list
            if output_shape[-2] is not None:
                output_shape[-2] *= self.scale_factor[0]
            if output_shape[-1] is not None:
                output_shape[-1] *= self.scale_factor[1]
            outshape.append(tuple(output_shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        a, b = self.scale_factor
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            upscaled = input
            if b > 1:
                upscaled = T.repeat_elements(upscaled, b, -1)
            if a > 1:
                upscaled = T.repeat_elements(upscaled, a, -2)
            outputs.append(upscaled)
        self._log_footprint(training, inputs, outputs)
        return outputs


class GlobalPool(OdinFunction):

    """ Global pooling layer

    This layer pools globally across all trailing dimensions beyond the 2nd.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_function=T.mean, **kwargs):
        super(GlobalPool, self).__init__(
            incoming, unsupervised=False, **kwargs)
        self.pool_function = pool_function

    @property
    def output_shape(self):
        outshape = []
        for shape in self.input_shape:
            outshape.append(shape[:2])
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            if T.ndim(input) > 3:
                input = T.flatten(input, 3)
            outputs.append(self.pool_function(input, axis=2))
        self._log_footprint(training, inputs, outputs)
        return outputs
