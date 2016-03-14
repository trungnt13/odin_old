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
from .pool import UpSampling2D

__all__ = [
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

    Note
    ----
    flip_filters : bool (default: False) (NO more support in odin)
        Whether to flip the filters and perform a convolution, or not to flip
        them and perform a correlation. Flipping adds a bit of overhead, so it
        is disabled by default. In most cases this does not make a difference
        anyway because the filters are learnt. However, ``flip_filters`` should
        be set to ``True`` if weights are loaded into it that were learnt using
        a regular :class:`lasagne.layers.Conv2DLayer`, for example.
    """

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=T.np_glorot_uniform, b=T.np_constant,
                 nonlinearity=T.relu, flip_filters=True,
                 unsupervised=False,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(
            incoming, unsupervised=unsupervised, **kwargs)
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

    """ 2D convolutional layer

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

    def get_inv(self, incoming, **kwargs):
        if incoming is None:
            incoming = self.output_shape
        # ====== transpose and reverse filter ====== #
        W = T.dimshuffle(self.W, (1, 0, 2, 3))
        # reverse filter: [:, :, ::-1, ::-1]
        W = T.reverse(T.reverse(W, -1), axis=-2)
        num_filters = self.input_shape[0][1]
        # ====== pad ====== #
        if self.pad == 'valid' or self.pad == (0, 0):
            pad = 'full'
        elif self.pad == 'full':
            pad = 'valid'
        elif self.pad == 'same':
            pad = 'same'
        else:
            self.raise_runtime('Only support pad=valid, full, same, or (0,0).'
                               ' No support for pad={}'.format(self.pad))
        # ====== upsampling if stride > (1,1) ====== #
        if any(i > 1 for i in self.stride):
            incoming = UpSampling2D(incoming, scale_factor=self.stride)
        stride = (1, 1)
        # ====== deconv ====== #
        b = None if self.b is None else T.np_constant
        nonlinearity = kwargs.get('nonlinearity', self.nonlinearity)
        deconv = Conv2D(incoming,
            num_filters=num_filters, filter_size=self.filter_size,
            stride=stride, pad=pad,
            untie_biases=self.untie_biases,
            W=W, b=b,
            nonlinearity=nonlinearity, **kwargs)
        # check number of channel match between deconv.input_shape and
        # conv.output_shape
        if deconv.input_shape[0][1] != self.output_shape[0][1]:
            self.raise_runtime('The number of channels in deconvolution '
                               'input_shape must equal to number of channels '
                               'in convolution output_shape, but '
                               'n_deconv_channels={} != n_conv_channels={}'
                               '.'.format(deconv.input_shape[0][1],
                                self.output_shape[0][1]))
        if deconv.output_shape[0][2:] != self.input_shape[0][2:]:
            self.raise_runtime('Deconvolution not work in this case, the '
                               'output_shape of deconvolution is different '
                               'from the input_shape of convolution, {} != '
                               ' {}'.format(deconv.output_shape[0][2:],
                                self.input_shape[0][2:]))
        return deconv


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
