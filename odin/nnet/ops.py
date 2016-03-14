from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_incoming_list

__all__ = [
    'Ops',
    'Get',
    'Flatten',
    'Reshape',
    'Dimshuffle',
    'Pad',
    'Inverse'
]


def pad(x, width, val=0, batch_ndim=1):
    """
    Pad a tensor with a constant value.

    Parameters
    ----------
    x : tensor

    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.

    val : float
        The constant value used for padding

    batch_ndim : integer
        Dimensions before the value will not be padded.

    """
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    if val:
        out = T.ones(output_shape) * val
    else:
        out = T.zeros(output_shape)
    return T.set_subtensor(out[tuple(indices)], x)


class Ops(OdinFunction):

    """
    This layer provides boilerplate for a custom layer that applies a
    simple transformation to the input.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    function : callable
        A function to be applied to the output of the previous layer.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    broadcast : bool (default:True)
        if True, broadcast the Operator to each input of this function,
        otherwise applying the function on the whole list of inputs

    Notes
    -----
    An :class:`ExpressionLayer` that does not change the shape of the data
    (i.e., is constructed with the default setting of ``output_shape=None``)
    is functionally equivalent to a :class:`NonlinearityLayer`.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, ExpressionLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ExpressionLayer(l_in, lambda X: X.mean(-1), output_shape='auto')
    >>> l1.output_shape
    (32, 100)
    """

    def __init__(self, incoming, ops, output_shape='auto',
                 broadcast=True, **kwargs):
        super(Ops, self).__init__(incoming, unsupervised=False, **kwargs)
        self.ops = ops
        self.broadcast = broadcast

        if isinstance(output_shape, (tuple, list)) and \
           isinstance(output_shape[-1], (int, float, long)):
            output_shape = [output_shape]
        self._shape = output_shape
        # calculate output_shape now
        self.output_shape

    @property
    def output_shape(self):
        if self._shape is None:
            return self.input_shape
        elif self._shape is 'auto':
            outshape = []
            inputs = []
            for shape in self.input_shape:
                shape = tuple(0 if s is None else s for s in shape)
                inputs.append(T.ones(shape))
            if self.broadcast:
                for i in inputs:
                    output_shape = T.eval(self.ops(i)).shape
                    outshape.append(tuple(s if s else None for s in output_shape))
            else: # merge operator
                output_shape = T.eval(T.shape(self.ops(inputs)))
                outshape.append(tuple(s if s else None for s in output_shape))
            return outshape
        else:
            return self._shape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        if self.broadcast:
            outputs = [self.ops(i) for i in inputs]
        else:
            outputs = self.ops(inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class Switch(object):

    """docstring for Switch"""

    def __init__(self, training, predicting):
        super(Switch, self).__init__()


class Get(OdinFunction):

    """Get a particular output at given indices and return"""

    def __init__(self, incoming, indices, **kwargs):
        super(Get, self).__init__(incoming, unsupervised=False, **kwargs)
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        # only OdinFunction can return multiple outputs
        n_inputs = self.n_inputs
        for i in indices:
            if i >= n_inputs:
                self.raise_arguments('Index can not be greater or equal to the '
                                     'number of inputs, in this case, index={} '
                                     '>= n_inputs={}'.format(i, n_inputs))
        self.indices = indices

    @property
    def output_shape(self):
        return [self.input_shape[i] for i in self.indices]

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = [inputs[i] for i in self.indices]
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class Flatten(OdinFunction):

    """
    A layer that flattens its input. The leading ``outdim-1`` dimensions of
    the output will have the same shape as the input. The remaining dimensions
    are collapsed into the last dimension.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    outdim : int
        The number of dimensions in the output.

    See Also
    --------
    flatten  : Shortcut
    """

    def __init__(self, incoming, outdim, unsupervised=False, **kwargs):
        super(Flatten, self).__init__(
            incoming, unsupervised=unsupervised, **kwargs)
        self.outdim = outdim

        if outdim < 1:
            self.raise_arguments('Dim must be >0, was %i', outdim)

    @property
    def output_shape(self):
        outshape = []
        for input_shape in self.input_shape:
            to_flatten = input_shape[self.outdim - 1:]

            if any(s is None for s in to_flatten):
                flattened = None
            else:
                flattened = int(np.prod(to_flatten))

            outshape.append(input_shape[:self.outdim - 1] + (flattened,))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            outputs.append(T.flatten(input, self.outdim))
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_inv(self, incoming, **kwargs):
        if incoming is None:
            incoming = self.output_shape
        shape = [-1 if i is None else i for i in self.input_shape[0]]
        inv = Reshape(incoming, shape=shape, **kwargs)
        for i, j in zip(inv.input_shape, self.output_shape):
            if i[1:] != j[1:]:
                self.raise_arguments('Inverted function incoming must have '
                                     'equal input_shape to output_shape of '
                                     'this function, but input_shape={} != '
                                     'output_shape={}'.format(
                                         i, j))
        return inv


class Reshape(OdinFunction):

    """
    A layer reshaping its input tensor to another tensor of the same total
    number of elements.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    shape : tuple
        The target shape specification. Each element can be one of:

        * ``i``, a positive integer directly giving the size of the dimension
        * ``[i]``, a single-element list of int, denoting to use the size
          of the ``i`` th input dimension
        * ``-1``, denoting to infer the size for this dimension to match
          the total number of elements in the input tensor (cannot be used
          more than once in a specification)
        * TensorVariable directly giving the size of the dimension

    Examples
    --------
    >>> from lasagne.layers import InputLayer, ReshapeLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ReshapeLayer(l_in, ((32, 50, 40)))
    >>> l1.output_shape
    (32, 50, 40)
    >>> l_in = InputLayer((None, 100, 20))
    >>> l1 = ReshapeLayer(l_in, ([0], [1], 5, -1))
    >>> l1.output_shape
    (None, 100, 5, 4)

    Notes
    -----
    The tensor elements will be fetched and placed in C-like order. That
    is, reshaping `[1,2,3,4,5,6]` to shape `(2,3)` will result in a matrix
    `[[1,2,3],[4,5,6]]`, not in `[[1,3,5],[2,4,6]]` (Fortran-like order),
    regardless of the memory layout of the input tensor. For C-contiguous
    input, reshaping is cheap, for others it may require copying the data.
    """

    def __init__(self, incoming, shape, unsupervised=False, **kwargs):
        super(Reshape, self).__init__(
            incoming, unsupervised=unsupervised, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < - 1:
                    raise ValueError("`shape` integers must be positive or -1")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise ValueError("`shape` input references must be "
                                     "single-element lists of int >= 0")
            elif T.is_expression(s):
                if T.ndim(s) != 0:
                    raise ValueError(
                        "A symbolic variable in a shape specification must be "
                        "a scalar, but had %i dimensions" % T.ndim(s))
            else:
                raise ValueError("`shape` must be a tuple of int and/or [int]")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape
        # try computing the output shape once as a sanity check
        self.output_shape

    @property
    def output_shape(self):
        # Initialize output shape from shape specification
        output_shape = list(self.shape)
        outshape = []
        for input_shape in self.input_shape:
            # First, replace all `[i]` with the corresponding input dimension, and
            # mask parts of the shapes thus becoming irrelevant for -1 inference
            masked_input_shape = list(input_shape)
            masked_output_shape = list(output_shape)
            for dim, o in enumerate(output_shape):
                if isinstance(o, list):
                    if o[0] >= len(input_shape):
                        raise ValueError("specification contains [%d], but input "
                                         "shape has %d dimensions only" %
                                         (o[0], len(input_shape)))
                    output_shape[dim] = input_shape[o[0]]
                    masked_output_shape[dim] = input_shape[o[0]]
                    if (input_shape[o[0]] is None) \
                       and (masked_input_shape[o[0]] is None):
                        # first time we copied this unknown input size: mask
                        # it, we have a 1:1 correspondence between out[dim] and
                        # in[o[0]] and can ignore it for -1 inference even if
                        # it is unknown.
                        masked_input_shape[o[0]] = 1
                        masked_output_shape[dim] = 1
            # Secondly, replace all symbolic shapes with `None`, as we cannot
            # infer their size here.
            for dim, o in enumerate(output_shape):
                if T.is_expression(o):
                    output_shape[dim] = None
                    masked_output_shape[dim] = None
            # From the shapes, compute the sizes of the input and output tensor
            input_size = (None if any(x is None for x in masked_input_shape)
                          else np.prod(masked_input_shape))
            output_size = (None if any(x is None for x in masked_output_shape)
                           else np.prod(masked_output_shape))
            del masked_input_shape, masked_output_shape
            # Finally, infer value for -1 if needed
            if -1 in output_shape:
                dim = output_shape.index(-1)
                if (input_size is None) or (output_size is None):
                    output_shape[dim] = None
                    output_size = None
                else:
                    output_size *= -1
                    output_shape[dim] = input_size // output_size
                    output_size *= output_shape[dim]
            # Sanity check
            if (input_size is not None) and (output_size is not None) \
               and (input_size != output_size):
                self.raise_runtime("%s cannot be reshaped to specification %s. "
                                 "The total size mismatches." %
                                 (input_shape, self.shape))
            outshape.append(tuple(output_shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        # Replace all `[i]` with the corresponding input dimension
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            output_shape = list(self.shape)
            for dim, o in enumerate(output_shape):
                if isinstance(o, list):
                    output_shape[dim] = T.shape(input)[o[0]]
            # Everything else is handled by Theano
            outputs.append(T.reshape(input, tuple(output_shape)))
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_inv(self, incoming, **kwargs):
        if incoming is None:
            incoming = self.output_shape
        shape = [-1 if i is None else i for i in self.input_shape[0]]
        inv = Reshape(incoming, shape=shape, **kwargs)
        for i, j in zip(inv.input_shape, self.output_shape):
            if i[1:] != j[1:]:
                self.raise_arguments('Inverted function incoming must have '
                                     'equal input_shape to output_shape of '
                                     'this function, but input_shape={} != '
                                     'output_shape={}'.format(
                                         i, j))
        return inv


class Dimshuffle(OdinFunction):

    """
    A layer that rearranges the dimension of its input tensor, maintaining
    the same same total number of elements.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape

    pattern : tuple
        The new dimension order, with each element giving the index
        of the dimension in the input tensor or `'x'` to broadcast it.
        For example `(3,2,1,0)` will reverse the order of a 4-dimensional
        tensor. Use `'x'` to broadcast, e.g. `(3,2,1,'x',0)` will
        take a 4 tensor of shape `(2,3,5,7)` as input and produce a
        tensor of shape `(7,5,3,1,2)` with the 4th dimension being
        broadcast-able. In general, all dimensions in the input tensor
        must be used to generate the output tensor. Omitting a dimension
        attempts to collapse it; this can only be done to broadcast-able
        dimensions, e.g. a 5-tensor of shape `(7,5,3,1,2)` with the 4th
        being broadcast-able can be shuffled with the pattern `(4,2,1,0)`
        collapsing the 4th dimension resulting in a tensor of shape
        `(2,3,5,7)`.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DimshuffleLayer
    >>> l_in = InputLayer((2, 3, 5, 7))
    >>> l1 = DimshuffleLayer(l_in, (3, 2, 1, 'x', 0))
    >>> l1.output_shape
    (7, 5, 3, 1, 2)
    >>> l2 = DimshuffleLayer(l1, (4, 2, 1, 0))
    >>> l2.output_shape
    (2, 3, 5, 7)
    """

    def __init__(self, incoming, pattern, unsupervised=False, **kwargs):
        super(Dimshuffle, self).__init__(incoming, unsupervised=unsupervised, **kwargs)

        # Sanity check the pattern
        used_dims = set()
        for p in pattern:
            if isinstance(p, int):
                # Dimension p
                if p in used_dims:
                    raise ValueError("pattern contains dimension {0} more "
                                     "than once".format(p))
                used_dims.add(p)
            elif p == 'x':
                # Broadcast
                pass
            else:
                raise ValueError("pattern should only contain dimension"
                                 "indices or 'x', not {0}".format(p))

        self.pattern = pattern

        # try computing the output shape once as a sanity check
        self.output_shape

    @property
    def output_shape(self):
        # Build output shape while keeping track of the dimensions that we are
        # attempting to collapse, so we can ensure that they are broadcastable
        outshape = []
        for input_shape in self.input_shape:
            output_shape = []
            dims_used = [False] * len(input_shape)
            for p in self.pattern:
                if isinstance(p, int):
                    if p < 0 or p >= len(input_shape):
                        raise ValueError("pattern contains {0}, but input shape "
                                         "has {1} dimensions "
                                         "only".format(p, len(input_shape)))
                    # Dimension p
                    o = input_shape[p]
                    dims_used[p] = True
                elif p == 'x':
                    # Broadcast; will be of size 1
                    o = 1
                output_shape.append(o)

            for i, (dim_size, used) in enumerate(zip(input_shape, dims_used)):
                if not used and dim_size != 1 and dim_size is not None:
                    raise ValueError(
                        "pattern attempted to collapse dimension "
                        "{0} of size {1}; dimensions with size != 1/None are not"
                        "broadcastable and cannot be "
                        "collapsed".format(i, dim_size))

            outshape.append(tuple(output_shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input, shape in zip(inputs, self.input_shape):
            # add broadcast in case we drop broadcastable dimension
            if len(self.pattern) < len(shape):
                drop_axes = [i for i in range(len(shape))
                             if i not in self.pattern and shape[i] == 1]
                if len(drop_axes) > 0:
                    input = T.addbroadcast(input, *drop_axes)
            # dimshuffle
            outputs.append(T.dimshuffle(input, self.pattern))
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_inv(self, incoming, **kwargs):
        if incoming is None:
            incoming = self.output_shape

        # remove the 'x'
        pattern = [i for i in self.pattern if i != 'x']
        # find the index of each element in pattern to revert the dimshuffle
        orig_pattern = tuple([self.pattern.index(i)
                              for i in range(len(pattern))])

        inv = Dimshuffle(incoming, pattern=orig_pattern, **kwargs)
        for i, j in zip(inv.input_shape, self.output_shape):
            if i[1:] != j[1:]:
                self.raise_arguments('Inverted function incoming must have '
                                     'equal input_shape to output_shape of '
                                     'this function, but input_shape={} != '
                                     'output_shape={}'.format(
                                         i, j))
        for i, j in zip(inv.output_shape, self.input_shape):
            if i[1:] != j[1:]:
                self.raise_arguments('Invert dimshuffle will not work in '
                                     'the case some dimension was removed.')
        return inv


class Pad(OdinFunction):

    """
    Pad all dimensions except the first ``batch_ndim`` with ``width``
    zeros on both sides, or with another value specified in ``val``.
    Individual padding for each dimension or edge can be specified
    using a tuple or list of tuples for ``width``.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.

    val : float
        Value used for padding

    batch_ndim : int
        Dimensions up to this value are not padded. For padding convolutional
        layers this should be set to 2 so the sample and filter dimensions are
        not padded
    """

    def __init__(self, incoming, width, batch_ndim=2, val=0, **kwargs):
        super(Pad, self).__init__(incoming, unsupervised=False, **kwargs)
        self.width = width
        self.val = val
        self.batch_ndim = batch_ndim

    @property
    def output_shape(self):
        outshape = []
        for input_shape in self.input_shape:
            output_shape = list(input_shape)

            if isinstance(self.width, int):
                widths = [self.width] * (len(input_shape) - self.batch_ndim)
            else:
                widths = self.width

            for k, w in enumerate(widths):
                if output_shape[k + self.batch_ndim] is None:
                    continue
                else:
                    try:
                        l, r = w
                    except TypeError:
                        l = r = w
                    output_shape[k + self.batch_ndim] += l + r
            outshape.append(tuple(output_shape))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            outputs.append(pad(input, self.width, self.val, self.batch_ndim))
        self._log_footprint(training, inputs, outputs)
        return outputs


class Inverse(OdinFunction):

    """
    The :class:`InverseLayer` class performs inverse operations
    for a single layer of a neural network by applying the
    partial derivative of the layer to be inverted with
    respect to its input: transposed layer
    for a :class:`DenseLayer`, deconvolutional layer for
    :class:`Conv2DLayer`, :class:`Conv1DLayer`; or
    an unpooling layer for :class:`MaxPool2DLayer`.

    It is specially useful for building (convolutional)
    autoencoders with tied parameters.

    Note that if the layer to be inverted contains a nonlinearity
    and/or a bias, the :class:`InverseLayer` will include the derivative
    of that in its computation.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    layer : a :class:`Layer` instance or a tuple
        The layer with respect to which the instance of the
        :class:`InverseLayer` is inverse to.

    Examples
    --------
    >>> import lasagne
    >>> from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer
    >>> from lasagne.layers import InverseLayer
    >>> l_in = InputLayer((100, 3, 28, 28))
    >>> l1 = Conv2DLayer(l_in, num_filters=16, filter_size=5)
    >>> l2 = DenseLayer(l1, num_units=20)
    >>> l_u = InverseLayer(l2, l1)  # As Deconv2DLayer
    """

    def __init__(self, incoming, function, unsupervised=None, **kwargs):
        if not isinstance(function, OdinFunction):
            self.raise_arguments('The function we want to take inverse must '
                                 'be OdinFunction, but its type is: {}'
                                 '.'.format(type(function)))

        if unsupervised is None:
            unsupervised = function.unsupervised
        super(Inverse, self).__init__(
            incoming, unsupervised=unsupervised, ** kwargs)
        if len(incoming.input_shape) != len(function.output_shape):
            self.raise_arguments('The number of input and output of function '
                                 'we invert must equal to the inputs to this '
                                 'Inverse function, but n_inverse={} != '
                                 'n_incoming={}'.format(
                                     len(function.output_shape),
                                     len(incoming.input_shape)))
        self.function = function

    @property
    def output_shape(self):
        return self.function.input_shape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        layer_out = self.function.get_cache_output(training)
        if layer_out is None: # function is not activated yet, activate it
            layer_out = self.function(training, **kwargs)
        # cached input so no disconnected graph
        layer_in = self.function.get_cache_input(training)
        if len(layer_in) > len(layer_out): # merge layer, multiple inputs - 1 output
            if len(layer_out) != 1:
                self.raise_arguments('Only support merge function with multiple'
                                     ' inputs and 1 outputs.')
            layer_in = [layer_in]
        elif len(layer_in) < len(layer_out): # 1 input multiple output
            if len(layer_in) != 1:
                self.raise_arguments('Only support multiplexer function with '
                                     'multiple outputs and 1 inputs.')
            layer_out = [layer_out]
        # ====== start invert process ====== #
        outputs = []
        for input, out_func, in_func in zip(inputs, layer_out, layer_in):
            if isinstance(out_func, (tuple, list)):
                known_grads = {i: input for i in out_func}
            else:
                known_grads = {out_func: input}
            outputs.append(
                T.gradients(None, in_func, known_grads=known_grads))
        self._log_footprint(training, inputs, outputs)
        return outputs
