# -*- coding: utf-8 -*-
# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

from six.moves import zip_longest, zip, range
import numpy as np
from collections import OrderedDict

from .. import tensor as T
from ..base import OdinFunction
from ..utils import as_index_map, as_incoming_list
from .dense import Dense
from .ops import Ops
from .normalization import BatchNormalization

__all__ = [
    "Gate",
    "Cell",
    "GatingCell",
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


def _create_dropout_mask(rng, p, shape, dtype):
    # ====== create dropout mask ====== #
    dropout_mask = None # shape: (batch_size, hidden_dims)
    if p is not None:
        # create dropout_mask and rescale it based on retain probability
        dropout_mask = rng.binomial(shape, p=p, dtype=dtype) / (1 - p)
    return dropout_mask

# ===========================================================================
# Step algorithm for GRU, LSTM, and be creative:
# Basic algorithm must have following arguments:
# hid_prev : tensor
#   previous hidden step, shape = (1, num_units)
# out_prev : list of tensor
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
                   gates_params, nonlinearity, slice_fn,
                   *const):
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
                  gates_params, nonlinearity, slice_fn,
                  *const):
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
                     gates_params, nonlinearity, slice_fn,
                     *const):
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
        self.name = name

        self.W_in = W_in
        self.W_hid = W_hid
        self.b = b
        self.W_cell = W_cell
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = T.linear
        else:
            self.nonlinearity = nonlinearity

    def __str__(self):
        return '<Gate:%s>' % self.name


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
    """

    def __init__(self, incoming,
                 nonlinearity=T.tanh,
                 memory=True,
                 batch_norm=False, learnable_norm=False,
                 dropoutW=None, dropoutU=None,
                 **kwargs):
        super(Cell, self).__init__(incoming, unsupervised=False, **kwargs)

        # input shape and number of cell units
        self.hidden_dims = self.input_shape[0][1:]
        for shape in self.input_shape:
            if self.hidden_dims != shape[1:]:
                self.raise_arguments('All incoming shape must have the same '
                                     'dimension, but {} != {}'
                                     '.'.format(shape, self.hidden_dims))

        # only setted when you stick this to Recurrent function
        self.input_dims = None

        # ====== check input_dims ====== #
        for i in self.incoming:
            if T.is_variable(i):
                self.set_learnable_incoming(i,
                    trainable=True, regularizable=False)
        # ====== W_cell and nonlinearity check ====== #
        if nonlinearity is None or not hasattr(nonlinearity, '__call__'):
            nonlinearity = T.linear
        self.nonlinearity = nonlinearity

        # ====== memory ====== #
        self.memory = memory

        # ====== batch_norm ====== #
        self.batch_norm = batch_norm
        self.batch_norm_dims = None # number of cell batch_norm handling
        self.learnable_norm = learnable_norm
        # ====== dropout ====== #
        if not hasattr(self, 'rng'):
            self.rng = T.rng(None)

        if dropoutW is not None and not T.is_variable(dropoutW) and \
           dropoutW > 0. and dropoutW < 1.:
            dropoutW = T.variable(dropoutW, name=self.name + '_dropoutW')
        else:
            dropoutW = None
        self.dropoutW = dropoutW

        if dropoutU is not None and not T.is_variable(dropoutU) and \
           dropoutU > 0. and dropoutU < 1.:
            dropoutU = T.variable(dropoutU, name=self.name + '_dropoutU')
        else:
            dropoutU = None
        self.dropoutU = dropoutU

    # ==================== methods for preparing parameters ==================== #
    def _check_batch_norm(self, norm_dims):
        '''BatchNormalization must be recreated if the number of gates changed,
        because for each gate we need 1 BatchNormalization
        '''
        if isinstance(norm_dims, (int, float, long)):
            norm_dims = (norm_dims,)

        # recreate batch_norm
        if self.batch_norm:
            # only re-new if number of gates changed
            if self.batch_norm_dims != norm_dims:
                self.log('Number of gates changed from {} to {}, hence, we '
                         'recreate BatchNormalization function for {} gates'
                         '.'.format(self.batch_norm_dims, norm_dims, norm_dims), 30)
                self.batch_norm_dims = norm_dims
                # check if gamma and beta are learnable
                if self.learnable_norm:
                    beta, gamma = T.np_constant, lambda x: T.np_constant(x, 1.)
                else:
                    beta, gamma = None, None
                # normalize over all batch and time dimension
                self.batch_norm = BatchNormalization(
                    (None, None,) + norm_dims,
                    axes=(0, 1),
                    beta=beta, gamma=gamma)
        else:
            self.batch_norm = None

    # ==================== Cell methods ==================== #
    def get_constants(self, X, Hinit, training, **kwargs):
        raise NotImplementedError

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
        raise NotImplementedError

    def step(self, in_precompute, hid_prev, out_prev, cell_prev, *constant):
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
        raise NotImplementedError

    # ==================== Override functions ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        if self.memory: # if has memory return all parameters
            params = super(Cell, self).get_params(
                globals, trainable, regularizable)
        else:
            # no memory, only return gates' parameters
            params = []
        # no safe check here, because the BatchNormalization used for
        # precompute can be different from the BatchNormalization from
        # get_params
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


class SimpleCell(Cell):

    """docstring for SimpleCell"""

    def __init__(self, arg):
        super(SimpleCell, self).__init__()
        self.arg = arg

    def precompute(self, X, training, **kwargs):
        # Because the input is given for all time steps, we can precompute
        # the inputs to hidden before scanning. First we need to reshape
        # from (seq_len, batch_size, trailing dimensions...) to
        # (seq_len*batch_size, trailing dimensions...)
        # This strange use of a generator in a tuple was because
        # input.shape[2:] was raising a Theano error
        # trailing_dims is features dimension
        seq_len, num_batch = T.shape(X)[0], T.shape(X)[1]

        trailing_dims = tuple(T.shape(X)[n] for n in range(2, T.ndim(X)))
        X = T.reshape(X, (seq_len * num_batch,) + trailing_dims)

        # Reshape back to (seq_len, batch_size, trailing dimensions...)
        _ = []
        for x in X:
            trailing_dims = tuple(T.shape(x)[n] for n in range(1, T.ndim(x)))
            _.append(T.reshape(x, (seq_len, num_batch) + trailing_dims))
        return X


class GatingCell(Cell):

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
    GatingCell now only support 2D hidden states
    """

    def __init__(self, incoming,
                 nonlinearity=T.tanh,
                 algorithm=simple_algorithm,
                 memory=True,
                 batch_norm=False, learnable_norm=False,
                 dropoutW=None, dropoutU=None,
                 **kwargs):
        super(GatingCell, self).__init__(incoming,
                 nonlinearity=nonlinearity,
                 memory=memory,
                 batch_norm=batch_norm, learnable_norm=learnable_norm,
                 dropoutW=dropoutW, dropoutU=dropoutU,
                 **kwargs)

        self.input_dims = None
        self.last_input_dims = self.input_dims

        # ====== check algorithm ====== #
        if algorithm is None:
            algorithm = simple_algorithm
        if not hasattr(algorithm, '__call__') or \
            algorithm.func_code.co_argcount < 8:
            self.raise_arguments('Algorithm function must be callable and '
                                 'has at least 8 input arguments, includes: '
                                 'hid_prev, out_prev, cell_prev, in_precompute, '
                                 'hid_precompute, gates_params, nonlinearity, '
                                 'and slice_function, to slice the concatenated'
                                 ' precomputed input and hidden.')
        self.algorithm = algorithm

        # store all gate informations
        self._gates = []
        # mapping: odin.nnet.rnn.Gate -> its compile params
        self._gates_params = OrderedDict()

    # ==================== Override functions ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        params = super(GatingCell, self).get_params(
            globals, trainable, regularizable)

        # ====== add gate params ====== #
        for g in self._gates_params.values():
            for i in g:
                if T.is_variable(i):
                    params += self.params[i.name].as_variables(
                        globals, trainable, regularizable)
        return params

    # ==================== methods for preparing parameters ==================== #
    def _check_gates(self):
        # name contain the ID of gate within the cells
        # name = str(len(self._gates)) if len(name) == 0 \
        #     else name + '_' + str(len(self._gates))
        if self.input_dims is None:
            self.raise_arguments('You must add this Cell to an '
                                 'odin.nnet.rnn.Recurrent function to infer '
                                 'the input_dims before doing any computation.')
        if set(self._gates) == set(self._gates_params.keys()) and \
           self.last_input_dims == self.input_dims:
            return
        self.last_input_dims = self.input_dims

        # ====== remove old params ====== #
        all_params = []
        for i in self._gates_params.values():
            all_params += [j for j in i if T.is_variable(j)]
        _ = OrderedDict()
        for i, j in self.params.iteritems():
            if j._params in all_params: # ignore old gate params
                continue
            _[i] = j
        self.params = _

        # ====== create new params ====== #
        num_inputs = np.prod(self.input_dims)
        num_units = np.prod(self.hidden_dims)
        self._gates_params = OrderedDict()
        for g in self._gates:
            name = g.name
            W_in = self.create_params(
                g.W_in, (num_inputs, num_units), 'W_in_' + str(name),
                regularizable=True, trainable=True)
            W_hid = self.create_params(
                g.W_hid, (num_units, num_units), 'W_hid_' + str(name),
                regularizable=True, trainable=True)
            # create cells
            if self.memory and g.W_cell is not None:
                W_cell = self.create_params(
                    g.W_cell, (num_units,), 'W_cell_' + str(name),
                    regularizable=True, trainable=True)
            else:
                W_cell = None

            if self.batch_norm: # no bias if batch_norm
                b = None
            else:
                b = self.create_params(
                    g.b, (num_units,), 'b_' + str(name),
                    regularizable=False, trainable=True)

            nonlinearity = g.nonlinearity
            if nonlinearity is None or not hasattr(nonlinearity, '__call__'):
                nonlinearity = T.linear
            # gate contain: W_in, W_hid, b, nonlinearity, W_cell (optional)
            self._gates_params[g] = [W_in, W_hid, b, nonlinearity, W_cell]

    def add_gate(self, gate):
        if not isinstance(gate, Gate):
            self.raise_arguments('gate must be instance of odin.nnet.rnn.Gate,'
                                 'but the given argument has type={}'
                                 '.'.format(type(gate)))
        self._gates.append(gate)
        return self

    def get_gate(self, name):
        ''' Return a list of gate had given name '''
        return [g for g in self._gates if g.name == name]

    # ==================== Cell methods ==================== #
    def _slice_w(self, W, n):
        num_units = np.prod(self.hidden_dims)
        if T.ndim(W) != 2:
            self.raise_arguments('Only slice weights with 2 dimensions.')
        return W[:, n * num_units:(n + 1) * num_units]

    def get_constants(self, X, Hinit, training, **kwargs):
        const = []
        # dropU
        shape = T.shape(Hinit)
        if self.dropoutU is not None:
            _ = []
            for i in range(len(self._gates)):
                _.append(
                    _create_dropout_mask(
                        self.rng, self.dropoutU, shape, Hinit.dtype))
            const.append(T.concatenate(_, axis=1))
        return const

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
        # initialization gate params
        self._check_gates()
        gates_params = [self._gates_params[g] for g in self._gates]
        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = T.concatenate(
            [i[0] for i in gates_params], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [i[1] for i in gates_params], axis=1)

        # Treat all dimensions after the second as flattened feature dimensions
        # it should be > 3 but as we reshaped X into
        # (num_batch * seq, trailing_dims) we only allow 2 dims.
        if T.ndim(X) > 3:
            X = T.flatten(X, 3)
        # applying dropout input with the same mask over all time step
        if self.dropoutW is not None:
            mask = self.rng.binomial(shape=T.shape(X)[1:], p=self.dropoutW,
                dtype=X.dtype)
            mask = T.expand_dims(mask, dim=0) # shape=(1, n_batch, n_features)
            X = X * mask # broadcast along time dimension

        # Because the input is given for all time steps, we can
        # precompute_input the inputs dot weight matrices before scanning.
        # W_in_stacked is (n_features, 4*num_units). input is then
        # (n_time_steps, n_batch, 4*num_units).
        self._check_batch_norm(np.prod(self.hidden_dims) * len(self._gates))
        if self.batch_norm: # apply batch normalization
            self.batch_norm.set_intermediate_inputs(
                T.dot(X, self.W_in_stacked), root=True)
            X = self.batch_norm(training, **kwargs)[0]
        else:
            # Stack biases into a (4*num_units) vector
            self.b_stacked = T.concatenate(
                [i[2] for i in gates_params], axis=0)
            X = T.dot(X, self.W_in_stacked) + self.b_stacked

        return X

    def step(self, in_precompute, hid_prev, out_prev, cell_prev, *constant):
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
        gates_params = [self._gates_params[g] for g in self._gates]
        # Extract the pre-activation gate values
        hid_precompute = T.dot(hid_prev, self.W_hid_stacked)

        if len(constant) > 0:
            # applying dropout
            dropU = constant[0]
            hid_precompute = hid_precompute * dropU

        return self.algorithm(hid_prev, out_prev, cell_prev,
                              in_precompute, hid_precompute,
                              gates_params, self.nonlinearity, self._slice_w,
                              *constant)

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
    hidden_info : callable, np.ndarray, theano.shared or :class:`OdinFunction`,
                  variable, placeholder, shape tuple (i.e. placeholder)
        Initializer for initial hidden state (:math:`h_0`). The shape of the
        initialization must be `(1,) + hidden_to_hidden.output_shape[1:]`. In
        case, and :class:`OdinFunction` is given, the output_shape can be
        one initialization for all sample in batch (as given above) or
        `(batch_size,) + hidden_to_hidden.output_shape[1:]`
        If hidden_info is a variable, it is learnable.
    mask : a :class:`OdinFunction`, Lasagne :class:`Layer` instance, keras
           :class:`Models` instance, variable, placeholder or shape tuple
        Allows for a sequence mask to be input, for when sequences are of
        variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    hidden_to_output : :class:`OdinFunction`, None, int, shape tuple
        Function which transform the current hidden state to output of the
        funciton, the output will be passed during recurrent and can be used
        for generative model. This layer may be connected to a chain of layers.
        If an integer is given, it is the number of hidden unit and a
        :class:`odin.nnet.Dense` is created as default. Note: we only consider
        the first output from given :class:`OdinFunction`.
    output_init : callable, np.ndarray, theano.shared or :class:`OdinFunction`,
                  variable, placeholder, shape tuple (i.e. placeholder)
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
    dropout : None, float(0.0-1.0), varialbe, expression
        Fraction of the output to drop for after every recurrent connections.
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
    return_idx : int
        only return the given index of outputs from hidden_to_output connections
        because the hidden_to_output connections can return multiple outputs

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
        self.output_dims : always `list of shape tuple`, the shape of output
            excluded the batch_size

    We only have 1 hidden state but we can have multiple output in the recurrent

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
    .. [2] A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
           (http://arxiv.org/abs/1512.05287)

    """

    def __init__(self, incoming, hidden_info,
                 mask=None,
                 hidden_to_output=None, output_init=None,
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 return_idx=None,
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

        # ====== hidden info ====== #
        if hidden_info is None:
            self.raise_arguments('hidden_info must be specified to initilaize '
                                 'the internal hidden state. hidden_info can '
                                 'be: ndarray, variable, placeholder, '
                                 'OdinFunction. Anything that we can infer '
                                 'the shape for hidden_state from.')
        incoming_list.append(hidden_info) # hidden_info is the last incoming

        # ====== output_init ====== #
        if isinstance(hidden_to_output, OdinFunction):
            output_init = as_incoming_list(output_init)
            n_out = len(hidden_to_output.output_shape)
            # validate right number of output_init
            if len(output_init) == 1:
                output_init = output_init * n_out
            elif len(output_init) != n_out:
                self.raise_arguments('The number of output_init must equal '
                                     'to the number of outputs returned by '
                                     'hidden_to_output connection, but '
                                     'output_init={} != return_output={}'.format(
                                         len(output_init), n_out))
            # validate shape of each output_init
            tmp = []
            for init, shape in zip(output_init,
                                   hidden_to_output.output_shape):
                shape = [1 if i is None else i for i in shape]
                # None -> all default value is 0.
                if init is None:
                    init = T.zeros(shape=shape)
                # initializing funciton
                elif (hasattr(init, '__call__') and
                      not isinstance(init, OdinFunction)):
                    init = T.variable(init(shape=shape))
                tmp.append(init)
            incoming_list += tmp
            # check return idx
            if return_idx is not None:
                if not isinstance(return_idx, (tuple, list)):
                    return_idx = [return_idx]
                for i in return_idx:
                    if i < 0 or i >= n_out:
                        self.raise_arguments('return_idx can be None for return'
                                             ' all outputs from hidden_to_output'
                                             ' connection, or int represent '
                                             'specified index of output, we '
                                             'have {} outputs, but index={}'
                                             '.'.format(n_out, i))
        elif hidden_to_output is not None:
            self.raise_arguments('hidden_to_output connection can only be '
                                 'None-for no connection, or an OdinFunction '
                                 'for specified transformation, but '
                                 'type(hidden_to_output)=%s' %
                                 str(type(hidden_to_output)))
        self.return_idx = return_idx

        # ====== call the init of OdinFunction ====== #
        super(Recurrent, self).__init__(
            incoming_list, unsupervised=False, **kwargs)

        # ====== parameters ====== #
        self.precompute_input = True

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

        init_info = [hidden_info]
        if output_init is not None:
            init_info += output_init
        # don't need to worry about this, if i is variable it is always learnable
        for i in init_info:
            self.set_learnable_incoming(i, trainable=True, regularizable=False)

        if hidden_to_output is None:
            self.hidden_dims = tuple(self.input_shape[-1][1:])
            self.output_dims = [self.hidden_dims]
        else:
            n_out = len(hidden_to_output.output_shape)
            self.hidden_dims = tuple(self.input_shape[-(n_out + 1)][1:])
            self.output_dims = [tuple(i[1:]) for i in self.input_shape[-n_out:]]

        # ====== output to hidden ====== #
        # hidden_to_output can return multiple outputs
        if hidden_to_output is not None:
            # output_shape of hidden_to_output must equal to its output_init
            for output_shape, output_dims in zip(hidden_to_output.output_shape,
                                    self.output_dims):
                if output_shape[1:] != output_dims:
                    self.raise_arguments('output_init shape is different from '
                                         'the output_shape of hidden_to_output '
                                         'connection, output_dims={} != '
                                         'hidden_to_output.output={}'.format(
                                             output_dims, output_shape[1:]))
            # input_shape of hidden_to_output must equal to hidden_info shape
            # must get input_shape from the roots
            input_shape = []
            for i in hidden_to_output.get_roots():
                input_shape += i.input_shape
            for shape in input_shape:
                if shape[1:] != self.hidden_dims:
                    self.raise_arguments('hidden_info shape is different from '
                                         'the input_shape of hidden_to_output '
                                         'connection, hidden_dims={} != '
                                         'hidden_to_output.input={}'.format(
                                             self.hidden_dims, shape[1:]))

        self.hidden_to_output = hidden_to_output
        # store all cell added to this Recurrent function
        self._cells = []

    # ==================== Override methods of OdinFunction ==================== #
    def get_params(self, globals, trainable=None, regularizable=None):
        params = super(Recurrent, self).get_params(
            globals, trainable, regularizable)
        # ====== hidden_to_output params ====== #
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
        for shape in cell.input_shape:
            if not _check_shape_match(shape, (None,) + self.hidden_dims):
                self.raise_arguments('Cell output_shape must match the shape '
                                     'of hidden state, but '
                                     'cell.output_shape={} and '
                                     'hidden_dims={}'.format(
                                         shape, (None,) + self.hidden_dims))
        # check cell.input_dims is the same dimensions with self.input_dims
        if cell.input_dims is not None and cell.input_dims != self.input_dims:
            self.raise_arguments('Cell input_dims must be the same as input_dims'
                                 ' of this Recurrent function, but cell_dims={}'
                                 ' != recurrent_dims={}'.format(
                                     cell.input_dims, self.input_dims))
        cell.input_dims = self.input_dims
        # everything ok add the cell
        if cell in self._cells:
            self.raise_arguments('Cell cannot be duplicated in Recurrent function.')
        self._cells.append(cell)
        return self

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        # output_dims can be hidden_dims, or multple dims from hidden_to_output
        if self.hidden_to_output is None or self.return_idx is None:
            output_dims = self.output_dims
        else:
            output_dims = [self.output_dims[i] for i in self.return_idx]
        # ====== calculate the output_shape ====== #
        outshape = []
        input_shape = self.input_shape
        for i in self._incoming_mask[::2]:
            shape = input_shape[i]
            if self.only_return_final:
                outshape += [(shape[0],) + outdims
                             for outdims in output_dims]
            else:
                outshape += [(shape[0], shape[1],) + outdims
                             for outdims in output_dims]
        return outshape

    def __call__(self, training=False, **kwargs):
        # ====== prepare inputs ====== #
        inputs = self.get_input(training, **kwargs)
        outputs = []
        n_incoming = len(self._incoming_mask[::2])
        n_outputs = 0

        if self.hidden_to_output is None:
            hid_init = [inputs[-1]] * n_incoming
            hid_init_shape = self.input_shape[-1]
            out_init = [[]] * n_incoming
            out_init_shape = [None]
        else:
            n_outputs = len(self.hidden_to_output.output_shape)
            hid_init = [inputs[-(n_outputs + 1)]] * n_incoming
            hid_init_shape = self.input_shape[-(n_outputs + 1)]
            out_init = [inputs[-n_outputs:]] * n_incoming
            out_init_shape = self.input_shape[-n_outputs:]

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
            num_batch = T.shape(X)[1]
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
            if hid_init_shape[0] == 1:
                # in this case, the OdinFunction only return 1 hidden_init
                # vector, need to repeat it for each batch
                dot_dims = (list(range(1, T.ndim(Hinit) - 1)) +
                            [0, T.ndim(Hinit) - 1])
                Hinit = T.dot(T.ones((num_batch, 1)),
                              T.dimshuffle(Hinit, dot_dims))
            # we do the same for Oinit
            _ = []
            for outinit, outshape in zip(Oinit, out_init_shape):
                if outshape[0] == 1:
                    dot_dims = (list(range(1, T.ndim(outinit) - 1)) +
                                [0, T.ndim(outinit) - 1])
                    outinit = T.dot(T.ones((num_batch, 1)),
                                    T.dimshuffle(outinit, dot_dims))
                _.append(outinit)
            Oinit = _

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

            # ====== get const from cell ====== #
            non_sequences = []
            Cconst_map = {}
            for c in self._cells:
                const = [i
                         for i in c.get_constants(X, Hinit, training, **kwargs)
                         if i is not None]
                Cconst_map[c] = range(
                    len(non_sequences), len(non_sequences) + len(const))
                non_sequences += const

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
            outputs_info = [Hinit] + Oinit
            output_idx = list(range(
                hidden_idx + 1, hidden_idx + 1 + len(Oinit)))
            # cell_idx
            cell_idx = list(range(
                len(sequences) + len(outputs_info),
                len(sequences) + len(outputs_info) + len(Cinit)))
            outputs_info += Cinit
            const_idx = list(range(
                len(sequences) + len(outputs_info),
                len(sequences) + len(outputs_info) + len(non_sequences)))

            # print(input_idx, mask_idx, hidden_idx, output_idx, cell_idx, const_idx)
            # ====== Create single recurrent computation step function ====== #
            def step(*args):
                # preprocess arguments
                input_n = [args[i] for i in input_idx]
                mask = None if mask_idx is None else args[mask_idx]
                hid_prev = args[hidden_idx]
                out_prev = [args[i] for i in output_idx]
                cell_prev = [args[i] for i in cell_idx]
                const = [args[i] for i in const_idx]

                # variable for computation of each cell
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
                    # get constant value for cell
                    c_const = Cconst_map[c]
                    c_const = [const[i] for i in c_const]
                    # outputs are (hid_new, cell_new)
                    c = c.step(c_in, hid_prev, out_prev, c_prev, *c_const)
                    cell_to_hid.append(c[0])
                    cell_to_cell.append(c[1])
                # no None value in cell_to_cell
                cell_to_cell = [i for i in cell_to_cell if i is not None]
                if len(cell_to_cell) != len(cell_prev):
                    self.raise_arguments('The number of returned new cell states'
                                         ' is different from number of previous'
                                         ' cell states.')

                # cell calculation give news hidden states
                if len(cell_to_hid) > 1:
                    hid = sum(i for i in cell_to_hid) / len(cell_to_hid)
                elif len(cell_to_hid) == 1:
                    hid = cell_to_hid[0]
                else:
                    hid = hid_prev

                # Skip over any input with mask 0 by copying the previous
                # hidden state; proceed normally for any input with mask 1.
                if mask is not None:
                    hid = T.switch(mask, hid, hid_prev)
                    cell_to_cell = [T.switch(mask, c_new, c_prev)
                                    for c_prev, c_new in zip(cell_prev, cell_to_cell)]

                # not hidden_to_output connection is specified
                if len(out_prev) == 0:
                    return [hid] + cell_to_cell

                # Compute the hidden-to-output activation, we must go to the
                # roots and set the intermediate inputs
                self.hidden_to_output.set_intermediate_inputs(hid, root=True)
                out = self.hidden_to_output(training)

                return [hid] + out + cell_to_cell

            # ====== create loop or scan funciton ====== #
            if self.unroll_scan:
                # Explicitly unroll the recurrence instead of using scan
                out = T.loop(
                    step_fn=step,
                    sequences=sequences,
                    outputs_info=outputs_info,
                    non_sequences=non_sequences,
                    go_backwards=self.backwards,
                    n_steps=n_steps)[0]
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                out = T.scan(
                    step_fn=step,
                    sequences=sequences,
                    outputs_info=outputs_info,
                    non_sequences=non_sequences,
                    go_backwards=self.backwards,
                    truncate_gradient=self.gradient_steps)[0]

            # if hidden_to_output is not None, return the output instead of
            # hidden states
            if len(Oinit) > 0:
                out = out[1:1 + len(Oinit)] # return all output
                if self.return_idx is not None:
                    out = [out[i] for i in self.return_idx]
            elif isinstance(out, (tuple, list)): # in case return hid & cells
                out = out[0] # hidden
            # otherwise, only hidden state returned
            if not isinstance(out, (tuple, list)):
                out = [out]

            # When it is requested that we only return the final sequence step,
            # we need to slice it out immediately after scan is applied
            if self.only_return_final:
                out = [o[-1] for o in out]
            else:
                _ = []
                for o in out:
                    # dimshuffle back to (n_batch, n_time_steps, n_features))
                    o = T.dimshuffle(o, (1, 0,) + tuple(range(2, T.ndim(o))))
                    # if scan is backward reverse the output
                    if self.backwards:
                        o = o[:, ::-1]
                    _.append(o)
                out = _
            outputs += out
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


# ===========================================================================
# Particular architecture
# ===========================================================================
class GRU(Recurrent):

    """Conventional GRU"""

    def __init__(self, incoming, hidden_info, mask=None,
                 resetgate=Gate(),
                 updategate=Gate(),
                 hidden_update=Gate(nonlinearity=T.tanh),
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 batch_norm=False, learnable_norm=False,
                 dropoutW=None, dropoutU=None,
                 **kwargs):
        if isinstance(hidden_info, (int, long, float)):
            hidden_info = (1, int(hidden_info))
        if isinstance(hidden_info[-1], (int, long, float)):
            hidden_info = T.np_constant(hidden_info)

        super(GRU, self).__init__(incoming=incoming, mask=mask,
            hidden_info=hidden_info,
            hidden_to_output=None,
            backwards=backwards,
            unroll_scan=unroll_scan,
            only_return_final=only_return_final,
            **kwargs)
        # ====== create cell ====== #
        c = GatingCell((1,) + self.hidden_dims,
            nonlinearity=T.linear,
            algorithm=gru_algorithm, memory=False,
            batch_norm=batch_norm, learnable_norm=learnable_norm,
            dropoutW=dropoutW, dropoutU=dropoutU)
        updategate.name = 'updategate'
        resetgate.name = 'resetgate'
        hidden_update.name = 'hidden_update'
        c.add_gate(updategate)
        c.add_gate(resetgate)
        c.add_gate(hidden_update)
        self.add_cell(c)


class LSTM(Recurrent):

    """Conventional LSTM"""

    def __init__(self, incoming, hidden_info, mask=None,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=T.tanh),
                 outgate=Gate(),
                 cell_init=T.np_constant,
                 nonlinearity=T.tanh,
                 backwards=False,
                 unroll_scan=False,
                 only_return_final=False,
                 batch_norm=False, learnable_norm=False,
                 dropoutW=None, dropoutU=None,
                 **kwargs):
        if isinstance(hidden_info, (int, long, float)):
            hidden_info = (1, int(hidden_info))
        if isinstance(hidden_info[-1], (int, long, float)):
            hidden_info = T.np_constant(hidden_info)

        super(LSTM, self).__init__(incoming=incoming, mask=mask,
            hidden_info=hidden_info,
            hidden_to_output=None,
            backwards=backwards,
            unroll_scan=unroll_scan,
            only_return_final=only_return_final,
            **kwargs)
        # ====== create cell ====== #
        if hasattr(cell_init, '__call__') and \
           not isinstance(cell_init, OdinFunction):
            cell_init = cell_init(shape=(1,) + self.hidden_dims)
        c = GatingCell(cell_init,
            nonlinearity=nonlinearity,
            algorithm=lstm_algorithm, memory=True,
            batch_norm=batch_norm, learnable_norm=learnable_norm,
            dropoutW=dropoutW, dropoutU=dropoutU)
        ingate.name = 'ingate'
        forgetgate.name = 'forgetgate'
        cell.name = 'cell'
        outgate.name = 'outgate'
        c.add_gate(ingate)
        c.add_gate(forgetgate)
        c.add_gate(cell)
        c.add_gate(outgate)
        self.add_cell(c)
