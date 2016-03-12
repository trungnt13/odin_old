from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction

__all__ = [
    'Ops',
    'Get',
    'Reshape'
]


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

    def __init__(self, incoming, shape, **kwargs):
        super(Reshape, self).__init__(incoming, unsupervised=False, **kwargs)
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
