# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

import numpy as np

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple

__all__ = [
    "Pool2D",
    "MaxPool2D",
    "Pool3D",
    "MaxPool3D",
    "Conv2D",
    "Conv3D",
]


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int
        The output size corresponding to the given convolution parameters.

    Raises
    ------
    RuntimeError
        When an invalid padding is specified, a `RuntimeError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


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
        super(Pool2D, self).__init__(incoming, unsupervised=False, **kwargs)
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
        self.pad = as_tuple(pad, 2)
        self.mode = mode
        # The ignore_border argument is for compatibility with MaxPool2DLayer.
        # ignore_border=False is not supported. Borders are always ignored.
        if not ignore_border:
            raise NotImplementedError("Pool2D does not support "
                                      "ignore_border=False.")

    # ==================== abstract function ==================== #
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
            shape[3] = pool_output_length(i[3],
                                          pool_size=self.pool_size[1],
                                          stride=self.stride[1],
                                          pad=self.pad[1],
                                          ignore_border=True,
                                                 )
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
        super(Pool3D, self).__init__(incoming, unsupervised=False, **kwargs)
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


class BaseConvLayer(OdinFunction):

    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    n=None, **kwargs)

    Convolutional layer base class

    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=T.np_glorot_uniform, b=T.np_constant,
                 nonlinearity=T.relu, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, unsupervised=False, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = T.linear
        else:
            self.nonlinearity = nonlinearity

        # ====== Validate all inpus shape are the same ====== #
        i = self.input_shape[0][1:]
        for j in self.input_shape:
            if i != j[1:]:
                self.raise_arguments('All features dimensions of inputs must be'
                                     ' the same, %s is different from %s.' %
                                     (str(i), str(j[1:])))
        if n is None:
            n = len(self.input_shape[0]) - 2
        elif n != len(self.input_shape[0]) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape[0], n + 2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.W = self.create_params(W, self.get_W_shape(), name="W",
            regularizable=True, trainable=True)
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[0][2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.create_params(b, biases_shape, name="b",
                                    regularizable=False, trainable=True)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[0][1]
        return (self.num_filters, num_input_channels) + self.filter_size

    @property
    def output_shape(self):
        outshape = []
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n

        for i in self.input_shape:
            batchsize = i[0]
            outshape.append(((batchsize, self.num_filters) +
                    tuple(conv_output_length(input, filter, stride, p)
                          for input, filter, stride, p
                          in zip(i[2:], self.filter_size, self.stride, pad))))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for i in inputs:
            conved = self.convolve(i, **kwargs)
            if self.b is None:
                activation = conved
            elif self.untie_biases:
                activation = conved + T.expand_dims(self.b, 0)
            else:
                activation = conved + T.dimshuffle(self.b, ('x', 0) + ('x',) * self.n)
            outputs.append(self.nonlinearity(activation))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.

        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`

        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")

    def get_config(self):
        config = super(BaseConvLayer, self).get_config()
        config['W'] = self.W.shape.eval().tolist()
        if self.b is None:
            config['b'] = None
        else:
            config['b'] = self.b.shape.eval().tolist()
        config['n'] = self.n
        config['nonlinearity'] = self.nonlinearity.__name__
        config['num_filters'] = self.num_filters
        config['filter_size'] = self.filter_size
        config['stride'] = self.stride
        config['pad'] = self.pad
        config['untie_biases'] = self.untie_biases
        config['flip_filters'] = self.flip_filters
        return config


class Conv2D(BaseConvLayer):

    """
    lasagne.layers.Conv2DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
    **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This is an alternative implementation
    which uses ``theano.sandbox.cuda.dnn.dnn_conv`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False) (NO more support in odin)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learnt. However, ``flip_filters`` should
        be set to ``True`` if weights are loaded into it that were learnt using
        a regular :class:`lasagne.layers.Conv2DLayer`, for example.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.relu, **kwargs):
        super(Conv2D, self).__init__(incoming, num_filters,
                                     filter_size, stride, pad,
                                     untie_biases, W, b, nonlinearity,
                                     False, n=2, **kwargs)

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        conved = T.conv2d(input,
            kernel=self.W,
            strides=self.stride,
            border_mode=border_mode,
            dim_ordering='th')
        return conved


class Conv3D(BaseConvLayer):

    """
    lasagne.layers.Conv3DDNNLayer(incoming, num_filters, filter_size,
    stride=(1, 1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
    **kwargs)

    3D convolutional layer

    Performs a 3D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.  This implementation uses
    ``theano.sandbox.cuda.dnn.dnn_conv3d`` directly.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape ``(batch_size,
        num_input_channels, input_rows, input_columns, input_depth)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of three integers allows different symmetric
        padding per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape ``(num_filters,
        num_input_channels, filter_rows, filter_columns, filter_depth)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns, output_depth)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: False) (NO more support in odin)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learned, but if you want to compute
        predictions with pre-trained weights, take care if they need flipping.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    """

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.relu,
                 **kwargs):
        super(Conv3D, self).__init__(incoming, num_filters,
                                     filter_size, stride, pad,
                                     untie_biases, W, b, nonlinearity,
                                     False, n=3, **kwargs)
        if pad == 'same' and any(s % 2 == 0 for s in filter_size):
            self.raise_arguments('Conv3D only support even filter_size '
                                 'if pad="same".')

    def convolve(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        conved = T.conv3d(input,
                          kernel=self.W,
                          strides=self.stride,
                          border_mode=border_mode,
                          dim_ordering='th'
                        )
        return conved
