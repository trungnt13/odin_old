# -*- coding: utf-8 -*-
# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

from six.moves import zip_longest, zip, range
import numpy as np

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple
from .dense import Dense

__all__ = [
    "Cell",
    "Recurrent",
]


class Cell(OdinFunction):

    """docstring for Cell"""

    def __init__(self, num_units, init=None, **kwargs):
        super(Cell, self).__init__(None, unsupervised=False, **kwargs)
        self.num_units = num_units
        self._gates = []

    def add_gate(self, W, b, nonlinearity):
        pass

    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training=False):
        raise NotImplementedError

    def step():
        pass

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        '''
        Parameters
        ----------
        objective : function
            often a function(y_pred, y_true) for supervised function, however,
            can have different form for unsupervised task
        optimizer : function (optional)
            function(loss_or_grads, params)
        globals : bool
            training on globals' parameters, or just optimize locals' parameters
        training : bool
            use output for training or output for prediction (in production)

        Return
        ------
        cost, updates : computational expression, OrderDict
            cost for monitoring the training process, and the update for the
            optimization
        '''
        raise NotImplementedError


def gru():
    pass


def block_input(x_t, h_t1, b):
    pass


class Recurrent(OdinFunction):

    """ Adapted implementation from Lasagne for Odin with many improvement
    A function which implements a recurrent connection with internal state.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-hidden connections by instantiating :class:`OdinFunction`
    instances and passing them on initialization.  Note that these connections
    can consist of multiple layers chained together.
    The output shape for the provided input-to-hidden and hidden-to-hidden
    connections must be the same.
    The output is computed by:

    .. math ::
        h_t = \sigma(f_i(x_t) + f_h(h_{t-1}))

    Parameters
    ----------
    incoming : a :class:`OdinFunction`, Lasagne :class:`Layer` instance, keras
               :class:`Models` instance, variable, placeholder or shape tuple
        The layer feeding into this layer, or the expected input shape.
    mask : a :class:`OdinFunction`, Lasagne :class:`Layer` instance, keras
           :class:`Models` instance, variable, placeholder or shape tuple
        Allows for a sequence mask to be input, for when sequences are of
        variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    hidden_to_hidden : int, :class:`OdinFunction`
        Function which transform the previous hidden state to the new state
        (:math:`f_h`). This layer may be connected to a chain of layers. If
        an integer is given, it is the number of hidden unit and a
        :class:`odin.nnet.Dense` is created as default. Note: we only consider
        the first output from given :class:`OdinFunction`.
    input_to_hidden : :class:`OdinFunction`
        :class:`OdinFunction` instance which transform input to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which has same input shape as `incoming`, except for the first
        dimension must be ``incoming.output_shape[0]*incoming.output_shape[1]``
        or ``None``. Note: we only consider the first output from given
        :class:`OdinFunction`.
    hidden_init : callable, np.ndarray, theano.shared or :class:`OdinFunction`,
                  variable, placeholder
        Initializer for initial hidden state (:math:`h_0`). The shape of the
        initialization must be `(1,) + hidden_to_hidden.output_shape[1:]`. In
        case, and :class:`OdinFunction` is given, the output_shape can be
        one initialization for all sample in batch (as given above) or
        `(batch_size,) + hidden_to_hidden.output_shape[1:]`
    learn_init : bool
        If True, initial hidden values are learned, which also means if
        `hidden_init` is an `OdinFunction`, all its parameters will be
        returned for training
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired). In this
        case, Theano makes an optimization which saves memory.

    Notes
    -----
    Default parameters which are fixed for convenient:
    gradient_steps : -1
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    precompute_input : True
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    Examples
    --------

    The following example constructs a simple `CustomRecurrentLayer` which
    has dense input-to-hidden and hidden-to-hidden connections.

    >>> import lasagne
    >>> n_batch, n_steps, n_in = (2, 3, 4)
    >>> n_hid = 5
    >>> l_in = lasagne.layers.InputLayer((n_batch, n_steps, n_in))
    >>> l_in_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_in)), n_hid)
    >>> l_hid_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_hid)), n_hid)
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid)

    The CustomRecurrentLayer can also support "convolutional recurrence", as is
    demonstrated below.

    >>> n_batch, n_steps, n_channels, width, height = (2, 3, 4, 5, 6)
    >>> n_out_filters = 7
    >>> filter_shape = (3, 3)
    >>> l_in = lasagne.layers.InputLayer(
    ...     (n_batch, n_steps, n_channels, width, height))
    >>> l_in_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer((None, n_channels, width, height)),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_hid_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer(l_in_to_hid.output_shape),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(
    ...     l_in, l_in_to_hid, l_hid_to_hid)

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """

    def __init__(self, incoming, mask=None,
                 hidden_to_hidden=None, input_to_hidden=None,
                 hidden_init=T.np_constant, learn_init=False,
                 nonlinearity=T.relu,
                 backwards=False,
                 grad_clipping=0,
                 unroll_scan=False,
                 only_return_final=False,
                 **kwargs):
        # ====== validate arguments ====== #
        if not isinstance(mask, (tuple, list)):
            mask = [mask]
        if not isinstance(incoming, (tuple, list)):
            incoming = [incoming]
        if isinstance(incoming[0], (int, float, long)): # shape tuple
            incoming = [incoming]
        # ====== process incoming ====== #
        self._incoming_mask = [] # list of [inc_idx, mask_idx, inc_idx, ...]
        incoming_list = []
        for i, j in zip_longest(incoming, mask):
            incoming_list.append(i)
            self._incoming_mask.append(len(incoming_list) - 1)
            if j is not None:
                incoming_list.append(j)
                self._incoming_mask.append(len(incoming_list) - 1)
            else:
                self._incoming_mask.append(None)
        super(Recurrent, self).__init__(
            incoming_list, unsupervised=False, **kwargs)
        # ====== parameters ====== #
        self.precompute_input = True
        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden

        self.backwards = backwards
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan

        self.gradient_steps = -1
        self.only_return_final = only_return_final
        # ====== validate input_dim ====== #
        self.input_dims = self.input_shape[0][2:]
        for i in self._incoming_mask[::2]: # incoming is the second
            if self.input_shape[i][2:] != self.input_dims:
                self.raise_arguments('All the input dimensions from the second '
                                     'dimension must be the same for all inputs,'
                                     ' %s != %s' % (str(self.input_shape[i][2:]),
                                     str(self.input_dims)))
        # ====== check hidden_to_hidden ====== #
        if isinstance(hidden_to_hidden, (int, long, float)):
            hidden_to_hidden = Dense((None, int(hidden_to_hidden)),
                num_units=int(hidden_to_hidden))
        elif not isinstance(hidden_to_hidden, OdinFunction):
            self.raise_arguments('hidden_to_hidden connection cannot be None, '
                                 'and must be int represent number of hidden '
                                 'units or a OdinFunction which transform '
                                 'hidden states at each step.')
        self.output_dims = hidden_to_hidden.output_shape[0][1:]
        if len(hidden_to_hidden.output_shape) > 1:
            self.raise_arguments('hidden_to_hidden connection should return '
                                 'only 1 output ')
        # ====== check input_to_hidden ====== #
        if input_to_hidden is None:
            input_to_hidden = Dense((None,) + self.input_dims,
                                    num_units=self.output_dims[0])
        elif isinstance(input_to_hidden, OdinFunction):
            pass
        else:
            self.raise_arguments('input_to_hidden connection only can be None, '
                                 'or OdinFunction which transform inputs at '
                                 'each step')
        if len(input_to_hidden.output_shape) > 1:
            self.raise_arguments('input_to_hidden connection should return '
                                 'only 1 output ')
        # ====== check more dimensions ====== #
        for i in self._incoming_mask[::2]:
            i = self.input_shape[i]
            if unroll_scan and i[1] is None:
                raise ValueError("Input sequence length cannot be specified as "
                                 "None when unroll_scan is True")
            # Check that the input_to_hidden connection can appropriately handle
            # a first dimension of input_shape[0]*input_shape[1] when we will
            # precompute the input dot product
            shape = input_to_hidden.output_shape[0]
            if (self.precompute_input and
                    shape[0] is not None and
                    i[0] is not None and
                    i[1] is not None and
                    (shape[0] != i[0] * i[1])):
                raise ValueError(
                    'When precompute_input == True, '
                    'input_to_hidden.output_shape[0] must equal '
                    'incoming.output_shape[0]*incoming.output_shape[1] '
                    '(i.e. batch_size*sequence_length) or be None but '
                    'input_to_hidden.output_shape[0] = {} and '
                    'incoming.output_shape[0]*incoming.output_shape[1] = '
                    '{}'.format(input_to_hidden.output_shape[0], i[0] * i[1]))

        # Check that input_to_hidden and hidden_to_hidden output shapes match,
        # but don't check a dimension if it's None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[0][1:],
                                     hidden_to_hidden.output_shape[0][1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.output_shape))

        # Check that input_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[0][1:],
                                     hidden_to_hidden.input_shape[0][1:])):
            raise ValueError("The output shape for input_to_hidden "
                             "must be equal to the input shape of "
                             "hidden_to_hidden after the first dimension, but "
                             "input_to_hidden.output_shape={} and "
                             "hidden_to_hidden.input_shape={}".format(
                                 input_to_hidden.output_shape,
                                 hidden_to_hidden.input_shape))

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        if nonlinearity is None:
            self.nonlinearity = T.linear
        else:
            self.nonlinearity = nonlinearity

        # ====== check hidden init ====== #
        if hidden_init is None:
            hidden_init = T.np_constant

        if T.is_variable(hidden_init) or T.is_expression(hidden_init):
            shape = tuple(T.eval(T.shape(hidden_init)))
        elif isinstance(hidden_init, OdinFunction):
            shape = tuple(hidden_init.output_shape[0][1:])
            if len(hidden_init.output_shape) != 1 and \
               len(hidden_init.output_shape) != len(incoming):
                self.raise_arguments('hidden_init must outputs 1 initialization '
                                     'for all incoming, or each initialization '
                                     'for each incoming, %d is wrong number of '
                                     'output.' % len(hidden_init.output_shape))
        else:
            shape = (1,) + self.output_dims
            hidden_init = self.create_params(hidden_init,
                shape=shape, name='hidden_init',
                regularizable=False, trainable=learn_init)()
        # hidden_init shape must same as output_dims
        if shape != (1,) + self.output_dims and \
           shape != self.output_dims:
            self.raise_arguments('hidden_init must have the same dimension '
                                 'with output_shape[0][1:] of hidden_to_hidden'
                                 ', but %s != %s' %
                                 (str(shape), str(self.output_dims)))
        self.hidden_init = hidden_init
        self.learn_init = learn_init

    # ==================== Override methods of OdinFunction ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        params = super(Recurrent, self).get_params(
            globals, trainable, regularizable)
        if isinstance(self.hidden_init, OdinFunction) and self.learn_init:
            params += self.hidden_init.get_params(
            globals, trainable, regularizable)
        params += self.input_to_hidden.get_params(
            globals, trainable, regularizable)
        params += self.hidden_to_hidden.get_params(
            globals, trainable, regularizable)
        return T.np_ordered_set(params).tolist()

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        outshape = []
        input_shape = self.input_shape
        for i in self._incoming_mask[::2]:
            shape = input_shape[i]
            if self.only_return_final:
                outshape.append((shape[0],) + self.output_dims)
            else:
                outshape.append((shape[0], shape[1],) + self.output_dims)
        return outshape

    def __call__(self, training=False):
        # ====== prepare inputs ====== #
        inputs = self.get_inputs(training)
        outputs = []
        if isinstance(self.hidden_init, OdinFunction):
            hid_init = self.hidden_init(training)
        else:
            hid_init = [self.hidden_init]
        if len(hid_init) == 1:
            hid_init = hid_init * len(self._incoming_mask[::2])

        # ====== create recurrent for each input ====== #
        for idx, (X, Xmask, Hinit) in enumerate(zip(self._incoming_mask[::2],
                                                self._incoming_mask[1::2],
                                                hid_init)):
            n_steps = self.input_shape[X][1]
            X = inputs[X]
            if Xmask is not None:
                Xmask = inputs[Xmask]

            # Input should be provided as (n_batch, n_time_steps, n_features)
            # but scan requires the iterable dimension to be first
            # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
            X = T.dimshuffle(X, (1, 0,) + tuple(range(2, T.ndim(X))))
            seq_len, num_batch = T.shape(X)[0], T.shape(X)[1]

            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(T.shape(X)[n] for n in range(2, T.ndim(X)))
            X = T.reshape(X, (seq_len * num_batch,) + trailing_dims)
            X = self.input_to_hidden.set_intermediate_inputs([X])(training)[0]

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(T.shape(X)[n] for n in range(1, T.ndim(X)))
            X = T.reshape(X, (seq_len, num_batch) + trailing_dims)

            # Create single recurrent computation step function
            def step(input_n, hid_previous, *args):
                # Compute the hidden-to-hidden activation
                hid_pre = self.hidden_to_hidden.set_intermediate_inputs(
                    [hid_previous])(training)[0]

                # If the dot product is precomputed then add it, otherwise
                # calculate the input_to_hidden values and add them
                hid_pre += input_n
                # Clip gradients
                if self.grad_clipping > 0:
                    hid_pre = T.grad_clip(hid_pre, self.grad_clipping)
                return self.nonlinearity(hid_pre)

            def step_masked(input_n, mask_n, hid_previous, *args):
                # Skip over any input with mask 0 by copying the previous
                # hidden state; proceed normally for any input with mask 1.
                hid = step(input_n, hid_previous, *args)
                hid_out = T.switch(mask_n, hid, hid_previous)
                return [hid_out]

            if Xmask is not None:
                Xmask = T.dimshuffle(Xmask, (1, 0, 'x'))
                sequences = [X, Xmask]
                step_fun = step_masked
            else:
                sequences = X
                step_fun = step

            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            if isinstance(self.hidden_init, OdinFunction) and \
               self.hidden_init.output_shape[idx][0] == 1:
                # in this case, the OdinFunction only return 1 hidden_init
                # vector, need to repeat it for each batch
                dot_dims = (list(range(1, T.ndim(Hinit) - 1)) +
                            [0, T.ndim(Hinit) - 1])
                Hinit = T.dot(T.ones((num_batch, 1)),
                              T.dimshuffle(Hinit, dot_dims))

            if self.unroll_scan:
                # Explicitly unroll the recurrence instead of using scan
                hid_out = T.loop(
                    step_fn=step_fun,
                    sequences=sequences,
                    outputs_info=[Hinit],
                    go_backwards=self.backwards,
                    n_steps=n_steps)[0]
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                hid_out = T.scan(
                    step_fn=step_fun,
                    sequences=sequences,
                    outputs_info=[Hinit],
                    go_backwards=self.backwards,
                    truncate_gradient=self.gradient_steps)[0]

            # When it is requested that we only return the final sequence step,
            # we need to slice it out immediately after scan is applied
            if self.only_return_final:
                hid_out = hid_out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                hid_out = T.dimshuffle(hid_out,
                    (1, 0,) + tuple(range(2, hid_out.ndim)))
                # if scan is backward reverse the output
                if self.backwards:
                    hid_out = hid_out[:, ::-1]
            outputs.append(hid_out)

        return outputs
