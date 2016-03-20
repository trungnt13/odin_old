# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# This module is created based on the code from keras library
# Original work Copyright (c) 2014-2015 keras contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

import numpy as np
import math

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple

__all__ = [
    "Pool2D",
    "MaxPool2D",
    "Pool3D",
    "MaxPool3D",
    'Unpool2DLayer',
    'UpSampling1D',
    'UpSampling2D',
    'GlobalPool'
]


def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


def unpooling_input(input, scale_factor, algorithm):
    if algorithm == 'pad':
        shape = list(T.shape(input))
        if scale_factor[-1] > 1:
            shape[-1] = shape[-1] * scale_factor[-1]
        if scale_factor[-2] > 1:
            shape[-2] = shape[-2] * scale_factor[-2]
        slices = (slice(None),) * (len(shape) - 2) + \
            (slice(None, None, scale_factor[-2]),
             slice(None, None, scale_factor[-1]))
        upsample = T.zeros(tuple(shape), dtype = input.dtype)
        upsample = T.set_subtensor(upsample[slices], input)
    # algorithm that repeat all axis of input
    elif algorithm == 'repeat':
        upsample = input
        if scale_factor[-1] > 1:
            upsample = T.repeat_elements(upsample, scale_factor[-1], -1)
        if scale_factor[-2] > 1:
            upsample = T.repeat_elements(upsample, scale_factor[-2], -2)

    return upsample


class Pool2D(OdinFunction):

    """
    2D pooling layer

    Performs 2D mean- or max-pooling over the two trailing axes of a 4D input
    tensor. This is an alternative implementation which uses
    ``theano.sandbox.cuda.dnn.dnn_pool`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension. If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool (default: True)
        This implementation never includes partial pooling regions, so this
        argument must always be set to True. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

    mode : string
        Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
        Defaults to 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    This is a drop-in replacement for :class:`lasagne.layers.MaxPool2DLayer`.
    Its interface is the same, except it does not support the ``ignore_border``
    argument.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2D, self).__init__(incoming, **kwargs)
        for i in self.input_shape:
            if len(i) != 4:
                raise ValueError("Tried to create a 2D pooling layer with "
                                 "input shape %r. Expected 4 input dimensions "
                                 "(batchsize, channels, 2 spatial dimensions)."
                                 % (i,))
        self.pool_size = as_tuple(pool_size, 2)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)
        if not isinstance(pad, str):
            pad = as_tuple(pad, 2)
        self.pad = pad
        self.mode = mode
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        if not ignore_border:
            raise NotImplementedError("Pool2D does not support "
                                      "ignore_border=False.")

    # ==================== abstract function ==================== #
    @property
    def output_shape(self):
        if self.pad == 'same':
            w_pad = self.pool_size[0] - 2 \
                if self.pool_size[0] % 2 == 1 else self.pool_size[0] - 1
            h_pad = self.pool_size[1] - 2 \
                if self.pool_size[1] % 2 == 1 else self.pool_size[1] - 1
            padding = (w_pad, h_pad)
        elif self.pad == 'valid':
            padding = (0, 0)
        else:
            padding = self.pad

        outshape = []
        for i in self.input_shape:
            shape = list(i)  # copy / convert to mutable list
            shape[2] = pool_output_length(i[2],
                                          pool_size=self.pool_size[0],
                                          stride=self.stride[0],
                                          pad=padding[0],
                                          ignore_border=True)
            shape[3] = pool_output_length(i[3],
                                          pool_size=self.pool_size[1],
                                          stride=self.stride[1],
                                          pad=padding[1],
                                          ignore_border=True)
            outshape.append(tuple(shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for i in inputs:
            outputs.append(
                T.pool2d(i, pool_size=self.pool_size, strides=self.stride,
                         pool_mode=self.mode, border_mode=self.pad,
                         dim_ordering='th')
            )
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class MaxPool2D(Pool2D):

    """
    2D max-pooling layer

    Subclass of :class:`Pool2DDNNLayer` fixing ``mode='max'``, provided for
    compatibility to other ``MaxPool2DLayer`` classes.
    """

    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0), ignore_border=True, **kwargs):
        super(MaxPool2D, self).__init__(incoming, pool_size, stride,
                                        pad, ignore_border, mode='max',
                                        **kwargs)


class Pool3D(OdinFunction):

    """
    3D pooling layer

    Performs 3D mean- or max-pooling over the 3 trailing axes of a 5D input
    tensor. This is an alternative implementation which uses
    ``theano.sandbox.cuda.dnn.dnn_pool`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension. If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool (default: True)
        This implementation never includes partial pooling regions, so this
        argument must always be set to True. It exists only to make sure the
        interface is compatible with :class:`lasagne.layers.MaxPool2DLayer`.

    mode : string
        Pooling mode, one of 'max', 'average_inc_pad' or 'average_exc_pad'.
        Defaults to 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool3D, self).__init__(incoming, **kwargs)
        for i in self.input_shape:
            if len(i) != 5:
                raise ValueError("Tried to create a 3D pooling layer with "
                                 "input shape %r. Expected 5 input dimensions "
                                 "(batchsize, channels, 3 spatial dimensions)."
                                 % (self.input_shape,))
        self.pool_size = as_tuple(pool_size, 3)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 3)
        self.pad = as_tuple(pad, 3)
        self.mode = mode
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        if not ignore_border:
            raise NotImplementedError("Pool3DDNNLayer does not support "
                                      "ignore_border=False.")

    @property
    def output_shape(self):
        outshape = []
        for i in self.input_shape:
            shape = list(i)  # copy / convert to mutable list
            shape[2] = pool_output_length(i[2],
                                          pool_size=self.pool_size[0],
                                          stride=self.stride[0],
                                          pad=self.pad[0],
                                          ignore_border=True,
                                        )
            shape[3] = pool_output_length(shape[3],
                                          pool_size=self.pool_size[1],
                                          stride=self.stride[1],
                                          pad=self.pad[1],
                                          ignore_border=True,
                                        )
            shape[4] = pool_output_length(shape[4],
                                          pool_size=self.pool_size[2],
                                          stride=self.stride[2],
                                          pad=self.pad[2],
                                          ignore_border=True,
                                        )
            outshape.append(tuple(shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for i in inputs:
            outputs.append(
                T.pool3d(i, pool_size=self.pool_size, strides=self.stride,
                         pool_mode=self.mode, border_mode=self.pad,
                         dim_ordering='th')
)
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class MaxPool3D(Pool3D):

    """
    3D max-pooling layer

    Subclass of :class:`Pool3DDNNLayer` fixing ``mode='max'``, provided for
    consistency to ``MaxPool2DLayer`` classes.
    """

    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0, 0), ignore_border=True, **kwargs):
        super(MaxPool3D, self).__init__(incoming, pool_size, stride,
                                        pad, ignore_border, mode='max',
                                        **kwargs)


class Unpool2DLayer(OdinFunction):

    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    ds is a tuple, denotes the upsampling
    algorithm : 'pad', 'repeat'
        'pad' is zero pad and set subtensor the value of input
        'repeat' is repeat the input then add zeros if don't have enough value
    """

    def __init__(self, incoming, scale_factor, algorithm='pad', **kwargs):
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if algorithm != 'pad' and algorithm != 'repeat':
            self.raise_arguments('Algorithm must equal to pad for padding zero'
                                 ' upsampling and equal to repeat for repeating'
                                 ' the input.')
        self.algorithm = algorithm

        scale_factor = as_tuple(scale_factor, 2, int)
        if scale_factor[0] < 1 or scale_factor[1] < 1:
            self.raise_arguments('Upscale factor must larger than 1')
        self.scale_factor = scale_factor

    @property
    def output_shape(self):
        outshape = []
        for input_shape in self.input_shape:
            output_shape = list(input_shape)
            output_shape[-2] = input_shape[-2] * self.scale_factor[-2]
            output_shape[-1] = input_shape[-1] * self.scale_factor[-1]
            outshape.append(tuple(output_shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            outputs.append(
                unpooling_input(input, self.scale_factor, self.algorithm))
        self._log_footprint(training, inputs, outputs)
        return outputs


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
        super(UpSampling1D, self).__init__(incoming, **kwargs)
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

    def __call__(self, training = False, **kwargs):
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
        super(UpSampling2D, self).__init__(incoming, **kwargs)

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

    def __call__(self, training = False, **kwargs):
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

    def __init__(self, incoming, pool_function = T.mean, **kwargs):
        super(GlobalPool, self).__init__(incoming, **kwargs)
        self.pool_function = pool_function

    @property
    def output_shape(self):
        outshape = []
        for shape in self.input_shape:
            outshape.append(shape[:2])
        return outshape

    def __call__(self, training = False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            if T.ndim(input) > 3:
                input = T.flatten(input, 3)
            outputs.append(self.pool_function(input, axis=2))
        self._log_footprint(training, inputs, outputs)
        return outputs
