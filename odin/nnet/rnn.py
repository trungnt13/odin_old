# -*- coding: utf-8 -*-
# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

from six.moves import zip_longest, zip, range
import numpy as np
from collections import defaultdict

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_tuple, as_index_map
from .dense import Dense
from .ops import Ops
from .normalization import BatchNormalization

__all__ = [
    "Gate",
    "Cell",
    "GRUCell",
    "Recurrent",
    "lstm_algorithm",
    "gru_algorithm",
    "simple_algorithm",
    "GRU",
    "LSTM",
]


# ===========================================================================
# Helper
# ===========================================================================
def _check_shape_match(shape1, shape2):
    ''' Check that shape1 is the same as shape2 but don't check a dimension
    if it's None for either shape

    Returns
    -------
    True : 2 given shapes are matched
    False : otherwise

    '''
    # TODO: (None, 32) will match (None, 32, 28, 28) - maybe an error
    if len(shape1) != len(shape2) or \
       not all(s1 is None or s2 is None or s1 == s2
               for s1, s2 in zip(shape1[1:], shape2[1:])):
        return False
    return True


def _validate_initialization(init, desire_dims, n_incoming,
    create_params, learnable, name):
    if init is None:
        init = T.np_constant

    if T.is_expression(init):
        shape = tuple(T.eval(T.shape(init)))
    elif isinstance(init, OdinFunction):
        shape = tuple(init.output_shape[0][1:])
        for i in init.output_shape:
            if shape != i[1:]:
                raise ValueError('For OdinFunction as initialization, all '
                                 'output_shape must have the same [1:] '
                                 'dimension.')
        if len(init.output_shape) != 1 and \
           len(init.output_shape) != n_incoming: # each incoming can have different init
            raise ValueError('initialization must outputs 1 initialization '
                             'for all incoming, or each initialization '
                             'for each incoming, %d is wrong number of '
                             'initialization.' % len(init.output_shape))
    else:
        shape = (1,) + desire_dims
        init = create_params(init, shape=shape, name=name,
            regularizable=False, trainable=learnable)
    # init shape must same as desire_dims (may or maynot include the batch_size)
    if shape != (1,) + desire_dims and \
       shape != desire_dims:
        raise ValueError('initialization must have the same dimension '
                         'with output_shape[0][1:] of given connection'
                         ', but %s != %s' %
                         (str(shape), str(desire_dims)))
    return init


# ===========================================================================
# Step algorithm for GRU, LSTM, and be creative:
# Basic algorithm must have following arguments:
# hid_prev : tensor
#   previous hidden step, shape = (1, num_units)
# out_prev : tensor
#   previous output (if no hidden_to_output connection specified, None),
#   shape = (1, output_dims)
# cell_prev : tensor
#   previous cell state (if cell is memoryless, cell_prev=None),
#   shape = (1, num_units)
# in_precompute : tensor
#   precomputed weighted input for all gates created by
#   T.dot(input_n, stack(W_in)), shape = (batch_size, num_units)
# hid_precompute : tensor
#   precomputed weighted hidden for all gates created by
#   T.dot(hidden_prev, stack(W_hid)), shape = (batch_size, num_units)
# gate_params : list
#   contain all information of cell parameters and its gates, each gate is a
#   list of 5 parameters [W_in, W_hid, b, nonlinearity, W_cell (if available)]
# nonlinearity : callable
#   activation function of this cell, apply on new hidden state
# slice_fn : function(W, n) -> W[n*size : (n+1)*size]
#   slice function of cell to slice the precomputed values for each gate.
# ===========================================================================
def lstm_algorithm(hid_prev, out_prev, cell_prev,
                   in_precompute, hid_precompute,
                   gates_params, nonlinearity, slice_fn):
    # For convention: the first 3 gates is:
    # input_gate, forget_gate and cell_new_input
    # all other the gates is output gates
    if len(gates_params) < 4:
        raise ValueError('LSTM algorithm need more than 4 gates to proceed. '
                         'By default, the first gate is input gate, then, '
                         'forget gate and cell_update. All the gates after '
                         'that will be applied as output gate to the new cell '
                         'state and sum up to new hidden state.')

    # sum of precomputed activated input and hidden_previous
    n_gates = len(gates_params)
    gates = in_precompute + hid_precompute
    gates = [slice_fn(gates, i) for i in range(n_gates)]

    # ====== not work ====== #
    # Compute cell-feedback connection to previous cell state
    for i in range(3):
        if gates_params[i][-1] is not None: # W_cell is at -1 index
            gates[i] = gates[i] + cell_prev * gates_params[i][-1]
        # Apply nonlinearities for incoming gates
        gates[i] = gates_params[i][-2](gates[i])

    # calculate new cells: c_t = f_t * c_prev + i_t * c_in
    cell = gates[1] * cell_prev + gates[0] * gates[2]

    hid = None
    for i in range(3, len(gates)):
        if gates_params[i][-1] is not None: # W_cell is at -1 index
            gates[i] = gates[i] + cell * gates_params[i][-1]
        # Apply nonlinearities for incoming gates
        gates[i] = gates_params[i][-2](gates[i])

        if hid is None:
            hid = gates[i] * nonlinearity(cell)
        else:
            hid = hid + gates[i] * nonlinearity(cell)
    if len(gates) - 3 > 1:
        hid = hid / (len(gates) - 3)
    return hid, cell


def gru_algorithm(hid_prev, out_prev, cell_prev,
                  in_precompute, hid_precompute,
                  gates_params, nonlinearity, slice_fn):
    if len(gates_params) < 3:
        raise ValueError('GRU algorithm need more than 3 gates to proceed. '
                         'By default, forget and hidden_update are the last '
                         '2 gates, and all the gates before them are update '
                         'gates. Hence, we can have infinite number of update '
                         'gates and they will be summed up to give final new '
                         'hidden state.')
    # ====== update gate ====== #
    n_gates = len(gates_params)
    update_gates = []
    for i, g in enumerate(gates_params[:-2]):
        gate = slice_fn(in_precompute, i) + slice_fn(hid_precompute, i)
        # activate nonlinearity
        gate = g[-2](gate)
        update_gates.append(gate)

    # ====== reset gate ====== #
    reset_gate = slice_fn(in_precompute, n_gates - 2) + slice_fn(hid_precompute, n_gates - 2)
    # activate nonlinearity
    reset_gate = gates_params[-2][-2](reset_gate)

    # ====== hidden update ====== #
    hid_update = slice_fn(in_precompute, n_gates - 1) + reset_gate * slice_fn(hid_precompute, n_gates - 1)
    hid_update = gates_params[-1][-2](hid_update)

    # ====== update new cell state ====== #
    hid = None
    for g in update_gates:
        if hid is None:
            hid = (1. - g) * hid_prev + g * hid_update
        else:
            hid = hid + (1. - g) * hid_prev + g * hid_update
    if len(update_gates) > 1:
        hid = hid / len(update_gates)
    return nonlinearity(hid), cell_prev


def simple_algorithm(hid_prev, out_prev, cell_prev,
                     in_precompute, hid_precompute,
                     gates_params, nonlinearity, slice_fn):
    ''' This algorithm take sum of all gated previous cell'''
    gates_precompute = in_precompute + hid_precompute
    # Compute cell-feedback connection to previous cell state
    # W_cell is at -1 index
    gates = []
    for i, p in enumerate(gates_params):
        g = slice_fn(gates_precompute, i)
        if p[-1] is not None:
            g = g + cell_prev * p[-1]
        gates.append(p[-2](g))

    if len(gates) > 0:
        cell = sum(cell_prev * i for i in gates) / len(gates)
    else:
        cell = cell_prev

    if cell is not None:
        hid = nonlinearity(cell)
    else:
        hid = hid_prev
    return hid, cell


# ===========================================================================
# Recurrent implementation
# ===========================================================================
class Gate(object):

    """ Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.

    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.

    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.

    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)

    References
    ----------
    .. [1] Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.

    """

    def __init__(self, W_in=T.np_glorot_normal, W_hid=T.np_glorot_normal,
                 W_cell=T.np_glorot_normal, b=T.np_constant,
                 nonlinearity=T.sigmoid,
                 name=''):
        self.W_in = W_in
        self.W_hid = W_hid
        self.b = b
        self.name = name
        self.W_cell = W_cell
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = T.linear
        else:
            self.nonlinearity = nonlinearity


class Cell(OdinFunction):

    """ Cell is a core memory with a collection of gates
    cell_init : shape tuple, int (num_units), variable, expression, `OdinFunction`
        the shape of cell_init must match the shape of internal hidden state of
        Recurrent function, it must has the shape (1, trailing_dims) or
        (batch_size, trailing_dims)
    input_dims : shape tuple, int
        Input shape to cell cannot contain None dimension, it must contain
        input_dims excluded `batch_size` and `seq_len` dimension.
    memory : bool
        if True this cell has its own memory vector which is internal state of
        the recurrent algorithm, otherwise, it only uses hidden state and
        output state (if available) as its memory

    Note
    ----
    Cell now only support 2D hidden states
    """

    def __init__(self, cell_init, input_dims, learnable=False,
                 nonlinearity=T.tanh,
                 algorithm=simple_algorithm,
                 batch_norm=False, learnable_norm=False,
                 memory=True,
                 **kwargs):
        super(Cell, self).__init__(cell_init, unsupervised=False, **kwargs)

        # input shape and number of cell units
        shape = self._validate_nD_input(2, strict=True)
        self.num_units = shape[-1]

        # ====== check input_dims ====== #
        if isinstance(input_dims, (int, float, long)):
            input_dims = (int(input_dims),)
        self.input_dims = input_dims
        if any(i is None for i in self.input_dims):
            self.raise_arguments('Input shape to cell cannot contain None'
                                 ' dimension, it must contain input_dims'
                                 ' excluded batch_size and seq_len dimension.')
        self.learnable = learnable
        for i in self.incoming:
            if T.is_variable(i):
                self.set_learnable_incoming(i,
                    trainable=learnable, regularizable=False)
        # ====== W_cell and nonlinearity check ====== #
        if nonlinearity is None or not hasattr(nonlinearity, '__call__'):
            nonlinearity = T.linear
        self.nonlinearity = nonlinearity
        # ====== check algorithm ====== #
        if algorithm is None:
            algorithm = simple_algorithm
        if not hasattr(algorithm, '__call__') or \
            algorithm.func_code.co_argcount != 8:
            self.raise_arguments('Algorithm function must be callable and '
                                 'has 8 input arguments, includes:hid_prev, '
                                 'out_prev, cell_prev, in_precompute, '
                                 'hid_precompute, gates_params, nonlinearity, '
                                 'and slice_function to slice the concatenated'
                                 ' precomputed input and hidden.')

        self.algorithm = algorithm
        self.memory = memory
        # store all gate informations
        self._gates = []
        self._gates_map = defaultdict(list) # store mapping name -> gate
        # ====== batch_norm ====== #
        self.batch_norm = batch_norm
        self.batch_norm_cell = 0 # number of cell batch_norm handling
        self.learnable_norm = learnable_norm

    def _check_batch_norm(self):
        '''BatchNormalization must be recreated if the number of gates changed,
        because for each gate we need 1 BatchNormalization
        '''
        # recreate batch_norm
        n_norm_gates = self.batch_norm_cell
        n_gates = len(self._gates)
        if n_gates == 0: # no gate no norm
            return

        if self.batch_norm:
            if n_norm_gates != n_gates:
                self.log('Number of gates changed from {} to {}, hence, we '
                         'recreate BatchNormalization function for {} gates'
                         '.'.format(n_norm_gates, n_gates, n_gates), 30)
            self.batch_norm_cell = n_gates
            # check if gamma and beta are learnable
            if self.learnable_norm:
                beta, gamma = T.np_constant, lambda x: T.np_constant(x, 1.)
            else:
                beta, gamma = None, None
            # normalize over all batch and time dimension
            self.batch_norm = BatchNormalization(
                (None, None, self.num_units * self.batch_norm_cell),
                axes=(0, 1),
                beta=beta, gamma=gamma)
        else:
            self.batch_norm = None

    def add_gate(self, W_in=T.np_glorot_normal,
                 W_hid=T.np_glorot_normal,
                 W_cell=T.np_glorot_normal,
                 b=T.np_constant,
                 nonlinearity=T.sigmoid,
                 name=''):
        # name contain the ID of gate within the cells
        # name = str(len(self._gates)) if len(name) == 0 \
        #     else name + '_' + str(len(self._gates))
        num_inputs = np.prod(self.input_dims)
        num_units = self.num_units

        W_in = self.create_params(
            W_in, (num_inputs, num_units), 'W_in_' + str(name),
            regularizable=True, trainable=True)
        W_hid = self.create_params(
            W_hid, (num_units, num_units), 'W_hid_' + str(name),
            regularizable=True, trainable=True)
        # create cells
        if self.memory and W_cell is not None:
            W_cell = self.create_params(
                W_cell, (num_units,), 'W_cell_' + str(name),
                regularizable=True, trainable=True)
        else:
            W_cell = None

        if self.batch_norm: # no bias if batch_norm
            b = None
        else:
            b = self.create_params(
                b, (num_units,), 'b_' + str(name),
                regularizable=False, trainable=True)

        if nonlinearity is None or not hasattr(nonlinearity, '__call__'):
            nonlinearity = T.linear
        # gate contain: W_in, W_hid, b, nonlinearity, W_cell (optional)
        gate = [W_in, W_hid, b, nonlinearity, W_cell]
        self._gates.append(gate)
        self._gates_map[name].append(gate)
        return self

    def get_gate(self, name):
        ''' Return a list of gate had given name '''
        return self._gates_map[name]

    # ==================== Cell methods ==================== #

    def _slice_w(self, W, n):
        if T.ndim(W) < 2:
            self.raise_arguments('Only slice weights with more than 2 '
                                 'dimensions.')
        return W[:, n * self.num_units:(n + 1) * self.num_units]

    def precompute(self, X, training, **kwargs):
        ''' We assume that the input X already preprocessed to have
        following dimension order: (n_time_steps, n_batch, n_features)

        Parameters
        ----------
        X : tensor
            (n_time_steps, n_batch, n_features)

        Return
        ------
        X : precomputed input

        '''
        if len(self._gates) == 0:
            return X

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = T.concatenate(
            [i[0] for i in self._gates], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [i[1] for i in self._gates], axis=1)

        # Treat all dimensions after the second as flattened feature dimensions
        if T.ndim(X) > 3:
            X = T.flatten(X, 3)

        # Because the input is given for all time steps, we can
        # precompute_input the inputs dot weight matrices before scanning.
        # W_in_stacked is (n_features, 4*num_units). input is then
        # (n_time_steps, n_batch, 4*num_units).
        self._check_batch_norm()
        if self.batch_norm: # apply batch normalization
            self.batch_norm.set_intermediate_inputs(
                T.dot(X, self.W_in_stacked), root=True)
            X = self.batch_norm(training, **kwargs)[0]
        else:
            # Stack biases into a (4*num_units) vector
            self.b_stacked = T.concatenate(
                [i[2] for i in self._gates], axis=0)
            X = T.dot(X, self.W_in_stacked) + self.b_stacked
        return X

    def step(self, in_precompute, hid_prev, out_prev, cell_prev):
        '''
        Parameters
        ----------
        in_precompute : tensor
            list of input tensors for this step function
        hid_prev : tensor
            preivious hidden state
        out_prev : tensor
            preivious output, in case no hidden_to_output connection specified,
            out_prev = hid_prev
        cell_prev : tensor
            previous cell state of this cell memory

        Returns
        -------
        (hid_new, cell_new) : list([tensor, tensor])
            for each input in given list, create a new hidden state and
            cell state
        '''
        if not hasattr(self, 'W_hid_stacked'):
            self.raise_arguments('Input to this step function must be the '
                                 'precompute version of input.')
        # Extract the pre-activation gate values
        hid_precompute = T.dot(hid_prev, self.W_hid_stacked)

        return self.algorithm(hid_prev, out_prev, cell_prev,
                              in_precompute, hid_precompute,
                              self._gates, self.nonlinearity, self._slice_w)

    # ==================== Override functions ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        if self.memory: # if has memory return all parameters
            params = super(Cell, self).get_params(
                globals, trainable, regularizable)
        else:
            # no memory, only return gates' parameters
            params = []
            for g in self._gates:
                for i in g:
                    if T.is_variable(i):
                        params += self.params[i.name].as_variables(
                            globals, trainable, regularizable)
        self._check_batch_norm()
        if isinstance(self.batch_norm, OdinFunction):
            params += self.batch_norm.get_params(
                globals, trainable, regularizable)
        return params

    @property
    def input_var(self):
        if self.memory:
            return super(Cell, self).input_var
        return []

    # ==================== Abstract function ==================== #
    @property
    def output_shape(self):
        if self.memory:
            return self.input_shape
        return []

    def __call__(self, training=False, **kwargs):
        ''' Return the initial states of cell '''
        if self.memory: # has memory cell, return its initial state
            inputs = self.get_input(training, **kwargs)
        else: # no memory
            inputs = []
        outputs = inputs
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class GRUCell(Cell):

    """GRUCell is a memoryless cell"""

    def __init__(self, hid_shape, input_dims,
                 nonlinearity=T.linear,
                 algorithm=gru_algorithm,
                 **kwargs):
        if isinstance(hid_shape, (int, float, long)):
            hid_shape = (None, int(hid_shape))
        elif isinstance(hid_shape, (tuple, list)) and \
        isinstance(hid_shape[-1], (int, float, long)):
            pass
        else:
            self.raise_arguments('hid_shape must be an integer - number of '
                                 'hidden units or shape tuple which is the '
                                 'shape of hidden state.')
        super(GRUCell, self).__init__(hid_shape, input_dims,
                 learnable=False, nonlinearity=nonlinearity,
                 algorithm=algorithm, memory=False,
                 **kwargs)


# ===========================================================================
# Main Recurrent algorithm
# ===========================================================================
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
    input_to_hidden : :class:`OdinFunction`, 'auto', None
        :class:`OdinFunction` instance which transform input to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which has same input shape as `incoming`, except for the first
        dimension must be ``incoming.output_shape[0]*incoming.output_shape[1]``
        or ``None``. Note: we only consider the first output from given
        :class:`OdinFunction`.
        If 'auto' is given, this layer automatically create a Dense layer to
        transform the input to the same dimension as hidden state
    hidden_to_output : :class:`OdinFunction`, None, int, shape tuple
        Function which transform the current hidden state to output of the
        funciton, the output will be passed during recurrent and can be used
        for generative model. This layer may be connected to a chain of layers.
        If an integer is given, it is the number of hidden unit and a
        :class:`odin.nnet.Dense` is created as default. Note: we only consider
        the first output from given :class:`OdinFunction`.
    hidden_init : callable, np.ndarray, theano.shared or :class:`OdinFunction`,
                  variable, placeholder
        Initializer for initial hidden state (:math:`h_0`). The shape of the
        initialization must be `(1,) + hidden_to_hidden.output_shape[1:]`. In
        case, and :class:`OdinFunction` is given, the output_shape can be
        one initialization for all sample in batch (as given above) or
        `(batch_size,) + hidden_to_hidden.output_shape[1:]`
    output_init : callable, np.ndarray, theano.shared or :class:`OdinFunction`,
                  variable, placeholder
        Initializer for initial output if the hidden_to_output connection is
        specified. The shape of the initialization must be
        `(1,) + hidden_to_hidden.output_shape[1:]`. In case, and
        :class:`OdinFunction` is given, the output_shape can be one
        initialization for all sample in batch (as given above) or
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
    In short, following rules must be applied:
        input_to_hidden.output_shape == hidden_to_hidden.input_shape
        hidden_to_hidden.input_shape == hidden_to_hidden.output_shape
        hidden_to_output.input_shape == hidden_to_hidden.output_shape

    Understand this variable would make it easier to implement new recurrent
    algorithm:
        self.input_dims : the input_shape excluded the first 2 dimensions of
            batch_size, seq_len
        self.hidden_dims : the shape of internal hidden state excluded the
            batch_size
        self.output_dims : the shape of output excluded the batch_size

    Default parameters which are fixed for convenient:
    gradient_steps : -1
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    precompute_input : True
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    grad_clipping : (removed, the way theano clip gradient is not effective)
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.

    Examples
    --------

    The following example constructs a simple `Recurrent` which
    has dense input-to-hidden and hidden-to-hidden connections.

    >>> import numpy as np
    >>> import odin
    >>> X = np.ones((128, 28, 10))
    >>> Xmask = np.ones((128, 28))
    ...
    >>> input_shape = (None, 28, 10)
    >>> mask_shape = (None, 28)
    >>> f = odin.nnet.Recurrent(
    ...     incoming=input_shape, mask=mask_shape,
    ...     input_to_hidden=odin.nnet.Dense((None, 10), num_units=13),
    ...     hidden_to_hidden=odin.nnet.Dense((None, 13), num_units=13))
    >>> f_pred = T.function(
    ...     inputs=f.input_var,
    ...     outputs=f())
    >>> print('Prediction shape:', [i.shape for i in f_pred(X, Xmask)])
    ... # Prediction shape: [(128, 28, 13)]

    The `Recurrent` can also support "convolutional recurrence", as is
    demonstrated below.

    >>> import numpy as np
    >>> import odin
    >>> n_batch, n_steps, n_channels, width, height = (13, 3, 4, 5, 6)
    >>> n_out_filters = 7
    >>> filter_shape = (3, 3)
    ...
    >>> X = np.random.rand(n_batch, n_steps, n_channels, width, height)
    ...
    >>> in_to_hid = odin.nnet.Conv2D(
    ...     incoming=(None, n_channels, width, height),
    ...     num_filters=n_out_filters, filter_size=filter_shape, pad='same')
    >>> hid_to_hid = odin.nnet.Conv2D(
    ...     incoming = in_to_hid.output_shape,
    ...     num_filters=n_out_filters, filter_size=filter_shape, pad='same')
    >>> f = odin.nnet.Recurrent(
    ...     incoming=(n_batch, n_steps, n_channels, width, height),
    ...     input_to_hidden=in_to_hid,
    ...     hidden_to_hidden=hid_to_hid)
    >>> f_pred = T.function(
    ...     inputs=f.input_var,
    ...     outputs=f())
    >>> print('Prediction shape:', [i.shape for i in f_pred(X)])
    ... # Prediction shape: [(13, 3, 7, 5, 6)]

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """

    def __init__(self, incoming, mask=None,
                 hidden_to_hidden=None,
                 input_to_hidden=None,
                 hidden_to_output=None,
                 hidden_init=T.np_constant,
                 output_init=T.np_constant,
                 learn_init=False,
                 nonlinearity=T.relu,
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 **kwargs):

        # ====== validate arguments ====== #
        if not isinstance(mask, (tuple, list)) or \
           isinstance(mask[-1], (int, float, long)):
            mask = [mask]
        # shape tuple
        if not isinstance(incoming, (tuple, list)) or \
           isinstance(incoming[-1], (int, float, long)):
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
        self.unroll_scan = unroll_scan

        self.gradient_steps = -1
        self.only_return_final = only_return_final

        # ====== validate input_dims ====== #
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
                num_units=int(hidden_to_hidden), nonlinearity=T.linear)
        elif not isinstance(hidden_to_hidden, OdinFunction):
            self.raise_arguments('hidden_to_hidden connection cannot be None, '
                                 'and must be int represent number of hidden '
                                 'units or a OdinFunction which transform '
                                 'hidden states at each step.')
        self.hidden_dims = hidden_to_hidden.output_shape[0][1:]
        # hidden_to_hidden must only return 1 output
        if len(hidden_to_hidden.output_shape) > 1:
            self.raise_arguments('hidden_to_hidden connection should return '
                                 'only 1 output ')
        # must get input_shape from the roots
        hidden_to_hidden_input_shape = []
        for i in hidden_to_hidden.get_roots():
            hidden_to_hidden_input_shape += i.input_shape
        # Check that hidden_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        output_shape = hidden_to_hidden.output_shape[0]
        for input_shape in hidden_to_hidden_input_shape:
            if not _check_shape_match(input_shape, output_shape):
                raise ValueError("The output shape for hidden_to_hidden "
                                 "must be equal to the input shape of "
                                 "hidden_to_hidden after the first dimension, "
                                 "but hidden_to_hidden.output_shape={} and "
                                 "hidden_to_hidden.input_shape={}".format(
                                     output_shape, input_shape))

        # ====== check input_to_hidden ====== #
        if input_to_hidden == 'auto':
            input_to_hidden = Dense((None,) + self.input_dims,
                                    num_units=np.prod(self.hidden_dims),
                                    nonlinearity=T.linear)
            if len(self.hidden_dims) > 1:
                input_to_hidden = Ops(input_to_hidden,
                    ops=lambda x: T.reshape(x, (-1,) + self.hidden_dims))
        elif isinstance(input_to_hidden, OdinFunction):
            pass
        elif input_to_hidden is None:
            pass
        else:
            self.raise_arguments('input_to_hidden connection only can be None, '
                                 'or an OdinFunction which transform inputs at '
                                 'each step')
        # validate the shape info of input_to_hidden
        if self.input_to_hidden is not None:
            # input_to_hidden must only return 1 output
            if len(input_to_hidden.output_shape) > 1:
                self.raise_arguments('input_to_hidden connection should return '
                                     'only 1 output ')

            # validate the match of input_to_hidden output_shape and input_shape
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
            if not _check_shape_match(input_to_hidden.output_shape[0],
                                      hidden_to_hidden.output_shape[0]):
                self.log("The output shape for input_to_hidden and "
                         "hidden_to_hidden must be equal after the first "
                         "dimension, but input_to_hidden.output_shape={} "
                         "and hidden_to_hidden.output_shape={}. Note:You might "
                         "use cell to transform the block input back to "
                         "hidden dimensions!".format(
                             input_to_hidden.output_shape,
                             hidden_to_hidden.output_shape), 30)

        # ====== output to hidden ====== #
        # if just shape tuple or an int, project the hidden to given dimensions
        if isinstance(hidden_to_output, (int, long, float)):
            hidden_to_output = (None, int(hidden_to_output))
        # create auto hidden_to_output given the shape
        if isinstance(hidden_to_output, (tuple, list)) and \
           isinstance(hidden_to_output[-1], (int, long, float)):
            # exclude first dimension which is the batch_size
            self.output_dims = hidden_to_output[1:]
            hidden_to_output = Dense((None,) + self.hidden_dims,
                num_units=np.prod(self.output_dims),
                nonlinearity=T.linear)
            if len(self.output_dims) > 1: # reshape to desire output dimension
                hidden_to_output = Ops(hidden_to_output,
                    ops=lambda x: T.reshape(x, (-1,) + self.output_dims))
        elif isinstance(hidden_to_output, OdinFunction):
            # exclude batch_size dimension
            self.output_dims = hidden_to_output.output_shape[0][1:]
        elif hidden_to_output is None:
            self.output_dims = self.hidden_dims
        else:
            self.raise_arguments('hidden_to_output connection only can be None, '
                                 'or an OdinFunction which transform hidden '
                                 'state at each step.')
        # validate shape of hidden_to_output if and OdinFunction is given
        if hidden_to_output is not None:
            # hidden_to_output must only return 1 output
            if len(hidden_to_output.output_shape) > 1:
                self.raise_arguments('hidden_to_output connection should return '
                                     'only 1 output.')
            # must get input_shape from the roots
            hidden_to_output_input_shape = []
            for i in hidden_to_output.get_roots():
                hidden_to_output_input_shape += i.input_shape
            # Check that hidden_to_output input_shape and hidden_to_hidden
            # output shapes match, but don't check a dimension if it's None
            # for either shape
            output_shape = hidden_to_hidden.output_shape[0]
            for input_shape in hidden_to_output_input_shape:
                if not _check_shape_match(input_shape, output_shape):
                    raise ValueError("The output shape for hidden_to_hidden and "
                                     "input shape for hidden_to_output must be "
                                     "equal after the first dimension, but "
                                     "hidden_to_hidden.output_shape={} and "
                                     "hidden_to_output.input_shape={}".format(
                                         output_shape, input_shape))

        # ====== Assign information ====== #
        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.hidden_to_output = hidden_to_output
        if nonlinearity is None:
            self.nonlinearity = T.linear
        else:
            self.nonlinearity = nonlinearity
        # ====== check hidden init ====== #
        self.hidden_init = _validate_initialization(
            hidden_init, self.hidden_dims, len(incoming),
            self.create_params, learn_init, name='hid_init')
        # ====== check output init ====== #
        if self.hidden_to_output is not None:
            output_init = _validate_initialization(
                output_init, self.output_dims, len(incoming),
                self.create_params, learn_init, name='out_init')
        self.output_init = output_init
        self.learn_init = learn_init
        # store all cell added to this Recurrent function
        self._cells = []

    # ==================== Override methods of OdinFunction ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        params = super(Recurrent, self).get_params(
            globals, trainable, regularizable)
        # ====== learn hidden_init and output_init ====== #
        if self.learn_init:
            # hidden_init
            if isinstance(self.hidden_init, OdinFunction):
                params += self.hidden_init.get_params(
                    globals, trainable, regularizable)
            elif T.is_variable(self.hidden_init):
                params.append(self.hidden_init)
            # output_init
            if isinstance(self.output_init, OdinFunction):
                params += self.output_init.get_params(
                    globals, trainable, regularizable)
            elif T.is_variable(self.output_init):
                params.append(self.output_init)
        # ====== connection ====== #
        if self.input_to_hidden is not None:
            params += self.input_to_hidden.get_params(
                globals, trainable, regularizable)
        params += self.hidden_to_hidden.get_params(
            globals, trainable, regularizable)
        if self.hidden_to_output is not None:
            params += self.hidden_to_output.get_params(
                globals, trainable, regularizable)
        # ====== cells ====== #
        for c in self._cells:
            params += c.get_params(globals, trainable, regularizable)
        return T.np_ordered_set(params).tolist()

    @property
    def input_var(self):
        placeholder = super(Recurrent, self).input_var
        for c in self._cells:
            placeholder += c.input_var
        return T.np_ordered_set(placeholder).tolist()

    # ==================== Recurrent methods ==================== #
    def add_cell(self, cell):
        if not isinstance(cell, Cell):
            self.raise_arguments('Only accept instance of odin.nnet.Cell.')
            return self
        # check cell.output_shape is the shape of hidden states,
        # input_shape = output_shape for cell, but GRUCell has no output_shape
        # hence, we check the input_shape
        cell_output_shape = cell.input_shape
        for i in cell_output_shape:
            if not _check_shape_match(i, (None,) + self.hidden_dims):
                self.raise_arguments('Cell output_shape must match the shape '
                                     'of hidden state, but '
                                     'cell.output_shape={} and '
                                     'hidden_dims={}'.format(
                                         i, (None,) + self.hidden_dims))
            if i[0] != 1 and i[0] is not None:
                self.raise_arguments('Cell initialization can only have first '
                                     'dimension equal to 1 or None.')
        # check cell.input_dims is the same dimensions with self.input_dims
        input_dims = self.input_dims
        if self.input_to_hidden is not None:
            input_dims = self.input_to_hidden.output_shape[0][1:]
        # np.prod because cell.input_dims can be flattened version
        if (np.prod(input_dims) != np.prod(cell.input_dims) and
            not _check_shape_match((None,) + cell.input_dims,
                                  (None,) + input_dims)):
            self.raise_arguments('Cell input_dims must match the input_dims '
                                 'of this Recurrent, or in the case '
                                 'input_to_hidden is not None, it must match '
                                 'the output_shape of input_to_hidden, but '
                                 'cell.input_dims={} and '
                                 'self.input_dims={}'.format(
                                 cell.input_dims, input_dims))
        # everything ok add the cell
        if cell in self._cells:
            self.raise_arguments('Cell cannot be duplicated in Recurrent function.')
        self._cells.append(cell)
        return self

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

    def __call__(self, training=False, **kwargs):
        # ====== prepare inputs ====== #
        inputs = self.get_input(training, **kwargs)
        outputs = []
        n_incoming = len(self._incoming_mask[::2])
        # roots can have multiple hierarchical, hence, we don't care about
        # OdinFunction, only need the input_variables
        n_hidden_to_hidden_inputs = sum(len([j for j in i.incoming
                                             if not isinstance(j, OdinFunction)])
                                        for i in self.hidden_to_hidden.get_roots())
        ## hidden init
        if isinstance(self.hidden_init, OdinFunction):
            # don't need to check number of returned values equal to 1,
            # or n_incoming, we already checked at _validate_initialization
            hid_init = self.hidden_init(training)
        else:
            hid_init = [self.hidden_init]
        if len(hid_init) == 1:
            hid_init = hid_init * n_incoming

        ## output init
        out_init = [None]
        if self.hidden_to_output is not None:
            if isinstance(self.output_init, OdinFunction):
                # don't need to check number of returned values equal to 1,
                # or n_incoming, we already checked at _validate_initialization
                out_init = self.output_init(training)
            else:
                out_init = [self.output_init]
        if len(out_init) == 1:
            out_init = out_init * n_incoming

        ## cell init
        cell_init = []
        for i in self._cells:
            i = i(training)
            # each cells will give number of initialization equal to number
            # of incoming
            if len(i) == 0:
                cell_init.append([None] * n_incoming)
            elif len(i) == 1:
                cell_init.append(i * n_incoming)
            elif len(i) == n_incoming:
                cell_init.append(i)
            else:
                self.raise_arguments('The number of initialization returned '
                                     'by cell can be 0, 1, or n_incoming, but '
                                     'n_cell_init={} and n_incoming={}.'.format(
                                         len(i), n_incoming))
        cell_init = [[j[i] for j in cell_init] for i in range(n_incoming)]

        # ====== create recurrent for each input ====== #
        for idx, (X, Xmask, Hinit, Oinit, Cinit) in enumerate(
            zip(self._incoming_mask[::2],
                self._incoming_mask[1::2],
                hid_init, out_init, cell_init)):
            n_steps = self.input_shape[X][1]
            X = inputs[X]
            if Xmask is not None:
                Xmask = inputs[Xmask]

            # Input should be provided as (n_batch, n_time_steps, n_features)
            # but scan requires the iterable dimension to be first
            # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
            X = T.dimshuffle(X, (1, 0,) + tuple(range(2, T.ndim(X))))
            seq_len, num_batch = T.shape(X)[0], T.shape(X)[1]

            if self.input_to_hidden is not None:
                # Because the input is given for all time steps, we can precompute
                # the inputs to hidden before scanning. First we need to reshape
                # from (seq_len, batch_size, trailing dimensions...) to
                # (seq_len*batch_size, trailing dimensions...)
                # This strange use of a generator in a tuple was because
                # input.shape[2:] was raising a Theano error
                # trailing_dims is features dimension
                trailing_dims = tuple(T.shape(X)[n] for n in range(2, T.ndim(X)))
                X = T.reshape(X, (seq_len * num_batch,) + trailing_dims)
                # call the input_to_hidden OdinFunction, we must go to the roots
                # and set the intermediate inputs
                self.input_to_hidden.set_intermediate_inputs(X, root=True)
                X = self.input_to_hidden(training)[0]
                # Reshape back to (seq_len, batch_size, trailing dimensions...)
                trailing_dims = tuple(T.shape(X)[n] for n in range(1, T.ndim(X)))
                X = T.reshape(X, (seq_len, num_batch) + trailing_dims)
            # calculate cell precompute_input
            if len(self._cells) > 0:
                X = [i.precompute(X, training, **kwargs) for i in self._cells]
            else:
                X = [X]
            # make sure no duplicate, and no None value
            X, Cinput_map = as_index_map(self._cells, X)

            # ====== check initialization of states (hidden, output) ====== #
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            if not isinstance(self.hidden_init, OdinFunction) or \
                (isinstance(self.hidden_init, OdinFunction) and
                 self.hidden_init.output_shape[idx][0] == 1):
                # in this case, the OdinFunction only return 1 hidden_init
                # vector, need to repeat it for each batch
                dot_dims = (list(range(1, T.ndim(Hinit) - 1)) +
                            [0, T.ndim(Hinit) - 1])
                Hinit = T.dot(T.ones((num_batch, 1)),
                              T.dimshuffle(Hinit, dot_dims))
            # we do the same for Oinit
            if self.hidden_to_output is not None:
                if not isinstance(self.output_init, OdinFunction) or \
                    (isinstance(self.output_init, OdinFunction) and
                     self.output_init.output_shape[idx][0] == 1):
                    dot_dims = (list(range(1, T.ndim(Oinit) - 1)) +
                                [0, T.ndim(Oinit) - 1])
                    Oinit = T.dot(T.ones((num_batch, 1)),
                                  T.dimshuffle(Oinit, dot_dims))
            # we do the same for Cinit
            tmp = []
            for i, j in zip(self._cells, Cinit):
                if j is None:
                    pass
                # the number of cell's output_shape maybe different from the
                # number of inputs to Recurrent
                elif i.output_shape[idx % len(i.output_shape)][0] == 1:
                    # in this case, the OdinFunction only return 1 hidden_init
                    # vector, need to repeat it for each batch
                    dot_dims = (list(range(1, T.ndim(j) - 1)) + [0, T.ndim(j) - 1])
                    j = T.dot(T.ones((num_batch, 1)), T.dimshuffle(j, dot_dims))
                tmp.append(j)
            # NO duplicated and None
            Cinit, Cinit_map = as_index_map(self._cells, tmp)

            # ====== check mask and form inputs to scan ====== #
            # only 1 input, 1 mask, 1 hidden, 1 output at a time
            # but cells can be multiple
            input_idx = list(range(len(X)))
            mask_idx = len(input_idx)
            sequences = []
            if Xmask is not None:
                Xmask = T.dimshuffle(Xmask, (1, 0, 'x'))
                sequences += X + [Xmask]
            else:
                sequences += X
                mask_idx = None
            hidden_idx = len(sequences)
            # output_idx
            outputs_info = [Hinit] if Oinit is None else [Hinit, Oinit]
            output_idx = hidden_idx + 1 if Oinit is not None else None
            # cell_idx
            cell_idx = len(sequences) + len(outputs_info)
            outputs_info += Cinit

            # ====== Create single recurrent computation step function ====== #
            def step(*args):
                # preprocess arguments
                input_n = [args[i] for i in input_idx]
                hid_prev = args[hidden_idx]
                out_prev = None
                if output_idx is not None:
                    out_prev = args[output_idx]
                cell_prev = [i for i in args[cell_idx:]]
                cell_to_hid = []
                cell_to_cell = []

                # TODO: decide to do hidden_to_hidden transform first or
                # calculating cell activation first

                # activate each cell one-by-one
                for c in self._cells:
                    # get input to cell based on Cell_input_map
                    c_in = Cinput_map[c]
                    if c_in is not None:
                        c_in = input_n[c_in]
                    # get cell previous state based on Cell_init_map
                    c_prev = Cinit_map[c]
                    if c_prev is not None:
                        c_prev = cell_prev[c_prev]
                    # outputs are (hid_new, cell_new)
                    c = c.step(c_in, hid_prev, out_prev, c_prev)
                    cell_to_hid.append(c[0])
                    cell_to_cell.append(c[1])
                # no None value in cell_to_cell
                cell_to_cell = [i for i in cell_to_cell if i is not None]
                if len(cell_to_cell) != len(cell_prev):
                    self.raise_arguments('The number of returned new cell states'
                                         ' is different from number of previous'
                                         ' cell states.')

                # cell calculation give news hidden states
                if len(cell_to_hid) > 0:
                    # if hidden_to_hidden accept multiple inputs
                    if n_hidden_to_hidden_inputs == len(cell_to_hid):
                        self.hidden_to_hidden.set_intermediate_inputs(
                            cell_to_hid, root=True)
                        hid = self.hidden_to_hidden(training)[0]
                    else: # fit each cell_hidden_state to hidden_to_hidden
                        hid = []
                        for h in cell_to_hid:
                            self.hidden_to_hidden.set_intermediate_inputs(
                                h, root=True)
                            hid.append(self.hidden_to_hidden(training)[0])
                        # naive approach, mean all of them
                        hid = sum(i for i in hid) / len(hid)
                else: # otherwise do normal recurrent calculation
                    # Compute the hidden-to-hidden activation, we must go to the
                    # roots and set the intermediate inputs
                    self.hidden_to_hidden.set_intermediate_inputs(
                        hid_prev, root=True)
                    hid = self.hidden_to_hidden(training)[0]

                    # If the dot product is precomputed then add it, otherwise
                    # calculate the input_to_hidden values and add them
                    for i in input_n:
                        hid = hid + i # plus transformed input
                    # activate hidden

                # final hidden state is activated one more time, you can use
                # T.linear to stop double nonlinear during recurrent
                hid = self.nonlinearity(hid)

                if mask_idx is not None:
                    # Skip over any input with mask 0 by copying the previous
                    # hidden state; proceed normally for any input with mask 1.
                    mask_n = args[mask_idx]
                    hid = T.switch(mask_n, hid, hid_prev)
                    cell_to_cell = [T.switch(mask_n, c_new, c_prev)
                                    for c_prev, c_new in zip(cell_prev, cell_to_cell)]

                # not hidden_to_output connection is specified
                if output_idx is None:
                    return [hid] + cell_to_cell

                # Compute the hidden-to-output activation, we must go to the
                # roots and set the intermediate inputs
                self.hidden_to_output.set_intermediate_inputs(hid, root=True)
                out = self.hidden_to_output(training)[0]

                return [hid, out] + cell_to_cell

            # ====== create loop or scan funciton ====== #
            if self.unroll_scan:
                # Explicitly unroll the recurrence instead of using scan
                out = T.loop(
                    step_fn=step,
                    sequences=sequences,
                    outputs_info=outputs_info,
                    go_backwards=self.backwards,
                    n_steps=n_steps)[0]
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                out = T.scan(
                    step_fn=step,
                    sequences=sequences,
                    outputs_info=outputs_info,
                    go_backwards=self.backwards,
                    truncate_gradient=self.gradient_steps)[0]

            # if hidden_to_output is not None, return the output instead of
            # hidden states
            if output_idx is not None:
                out = out[1] # output
            elif isinstance(out, (tuple, list)): # in case return hid & cells
                out = out[0] # hidden
            # otherwise, only hidden state returned

            # When it is requested that we only return the final sequence step,
            # we need to slice it out immediately after scan is applied
            if self.only_return_final:
                out = out[-1]
            else:
                # dimshuffle back to (n_batch, n_time_steps, n_features))
                out = T.dimshuffle(out,
                    (1, 0,) + tuple(range(2, T.ndim(out))))
                # if scan is backward reverse the output
                if self.backwards:
                    out = out[:, ::-1]
            outputs.append(out)
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


# ===========================================================================
# Particular architecture
# ===========================================================================
class GRU(Recurrent):

    """Conventional GRU"""

    def __init__(self, incoming, num_units, mask=None,
                 resetgate=Gate(),
                 updategate=Gate(),
                 hidden_update=Gate(nonlinearity=T.tanh),
                 hidden_init=T.np_constant, learn_init=False,
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 **kwargs):
        super(GRU, self).__init__(incoming=incoming, mask=mask,
            hidden_to_hidden=Ops(incoming=(None, num_units), ops=T.linear),
            input_to_hidden=None,
            hidden_to_output=None,
            output_init=None,
            hidden_init=hidden_init, learn_init=learn_init,
            nonlinearity=T.linear,
            backwards=backwards,
            unroll_scan=unroll_scan,
            only_return_final=only_return_final,
            **kwargs)
        # ====== create cell ====== #
        cell = GRUCell(hid_shape=(None, num_units), input_dims=self.input_dims,
                       nonlinearity=T.linear,
                       algorithm=gru_algorithm)
        cell.add_gate(W_in=updategate.W_in, W_hid=updategate.W_hid,
                      b=updategate.b, nonlinearity=updategate.nonlinearity,
                      name='update_gate')
        cell.add_gate(W_in=resetgate.W_in, W_hid=resetgate.W_hid,
                      b=resetgate.b, nonlinearity=resetgate.nonlinearity,
                      name='reset_gate')
        cell.add_gate(W_in=hidden_update.W_in, W_hid=hidden_update.W_hid,
                      b=hidden_update.b, nonlinearity=hidden_update.nonlinearity,
                      name='hidden_update')
        self.add_cell(cell)


class LSTM(Recurrent):

    """Conventional LSTM"""

    def __init__(self, incoming, num_units, mask=None,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=T.tanh),
                 outgate=Gate(),
                 cell_init=T.np_constant, hidden_init=T.np_constant,
                 learn_init=False,
                 nonlinearity=T.tanh,
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 **kwargs):
        super(LSTM, self).__init__(incoming=incoming, mask=mask,
            hidden_to_hidden=Ops(incoming=(None, num_units), ops=T.linear),
            input_to_hidden=None,
            hidden_to_output=None,
            output_init=None,
            hidden_init=hidden_init, learn_init=learn_init,
            nonlinearity=T.linear,
            backwards=backwards,
            unroll_scan=unroll_scan,
            only_return_final=only_return_final,
            **kwargs)
        # ====== validate cell_init ====== #
        if hasattr(cell_init, '__call__') and \
           not isinstance(cell_init, OdinFunction):
            cell_init = T.variable(cell_init(shape=(1, num_units)),
                name='cell_init')
        # ====== create cell ====== #
        c = Cell(cell_init=cell_init, input_dims=self.input_dims,
            learnable=learn_init, nonlinearity=nonlinearity,
            algorithm=lstm_algorithm, memory=True)
        c.add_gate(W_in=ingate.W_in, W_hid=ingate.W_hid,
                   W_cell=ingate.W_cell, b=ingate.b,
                   nonlinearity=ingate.nonlinearity,
                   name='input_gate')
        c.add_gate(W_in=forgetgate.W_in, W_hid=forgetgate.W_hid,
                   W_cell=forgetgate.W_cell, b=forgetgate.b,
                   nonlinearity=forgetgate.nonlinearity,
                   name='forget_gate')
        c.add_gate(W_in=cell.W_in, W_hid=cell.W_hid,
                   W_cell=cell.W_cell, b=cell.b,
                   nonlinearity=cell.nonlinearity,
                   name='cell')
        c.add_gate(W_in=outgate.W_in, W_hid=outgate.W_hid,
                   W_cell=outgate.W_cell, b=outgate.b,
                   nonlinearity=outgate.nonlinearity,
                   name='output_update')
        self.add_cell(c)
