from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from ..base import OdinFunction

__all__ = [
    'Ops',
    'Get'
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
                    output_shape = T.eval(T.shape(self.ops(i)))
                    outshape.append(tuple(s if s else None for s in output_shape))
            else: # merge operator
                output_shape = T.eval(T.shape(self.ops(inputs)))
                outshape.append(tuple(s if s else None for s in output_shape))
            return outshape
        else:
            return self._shape

    def __call__(self, training=False):
        inputs = self.get_inputs(training)
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

    """docstring for Get"""

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

    def __call__(self, training=False):
        inputs = self.get_inputs(training)
        outputs = [inputs[i] for i in self.indices]
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
