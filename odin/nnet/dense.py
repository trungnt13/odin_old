# ===========================================================================
# This module is adpated from 2 library: keras and lasagne
# Original work Copyright (c) 2014-2015 keras contributors
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import print_function, division, absolute_import


from .. import tensor as T
from ..base import OdinFunction
from .. import config

__all__ = [
    "Dense",
    "Embedding"
]


class Dense(OdinFunction):

    def __init__(self, incoming, num_units,
                 W=T.np_symmetric_uniform,
                 b=T.np_constant,
                 nonlinearity=T.relu,
                 **kwargs):
        super(Dense, self).__init__(
            incoming, **kwargs)

        num_inputs = self.input_shape[0][-1]
        for shape in self.input_shape:
            if shape[-1] != num_inputs:
                self.raise_arguments('The last dimensions of all inputs must '
                                     'equal.')
        shape = (num_inputs, num_units)

        self.W = self.create_params(
            W, shape, 'W', regularizable=True, trainable=True)
        if b is None:
            self.b = None
        else:
            self.b = self.create_params(
                b, (num_units,), 'b', regularizable=False, trainable=True)

        self.num_units = num_units
        self.num_inputs = num_inputs
        self.nonlinearity = (T.linear if nonlinearity is None else nonlinearity)

    @property
    def output_shape(self):
        return [tuple(i[:-1]) + (self.num_units,) for i in self.input_shape]

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        # ====== processing each inputs ====== #
        for x in inputs:
            # calculate projection
            activation = T.dot(x, self.W)
            if self.b is not None:
                activation = activation + self.b
            outputs.append(self.nonlinearity(activation))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_inv(self, incoming, **kwargs):
        W = T.transpose(self.W)
        b = None if self.b is None else T.np_constant
        nonlinearity = kwargs.pop('nonlinearity', self.nonlinearity)
        # auto create incoming
        if incoming is None:
            incoming = self.output_shape
        inv = Dense(incoming, num_units=self.num_inputs,
            W=W, b=b, nonlinearity=nonlinearity,
            **kwargs)
        # add reshape if Dense flatten the incoming
        if inv.num_inputs != self.num_units:
            self.raise_runtime('num_inputs of inverted function must equal to '
                               'num_units of this function, but '
                               'num_inputs={} != num_units={}'.format(
                                   inv.num_inputs, self.num_units
                               ))
        return inv


class VariationalDense(OdinFunction):

    def __init__(self, incoming, num_units,
                 W=T.np_symmetric_uniform,
                 b=T.np_constant,
                 nonlinearity=T.relu,
                 prior_mean=0.,
                 prior_std=T.exp(-3),
                 regularizer_scale=1.,
                 **kwargs):
        super(VariationalDense, self).__init__(
            incoming, **kwargs)
        if self.rng is None:
            self.rng = T.rng(None)

        # ====== check shape ====== #
        num_inputs = self.input_shape[0][-1]
        for shape in self.input_shape:
            if shape[-1] != num_inputs:
                self.raise_arguments('The last dimensions of all inputs must '
                                     'equal.')
        shape = (num_inputs, num_units)
        # ====== create params ====== #
        self.W_mu = self.create_params(W, shape=shape, name="W_mu",
            regularizable=True, trainable=True)
        self.W_logsigma = self.add_param(W, shape=shape, name="W_logsigma",
            regularizable=True, training=True)
        if b is None:
            self.b_mu = None
            self.b_logsigma = None
        else:
            self.b_mu = self.create_params(
                b, (num_units,), 'b', regularizable=False, trainable=True)
            self.b_logsigma = self.create_params(
                b, (num_units,), 'b', regularizable=False, trainable=True)
        # ====== cache ====== #
        self.W_cache = None
        self.b_cache = None
        # ====== attribute ====== #
        self.nonlinearity = (T.linear if nonlinearity is None else nonlinearity)
        self.prior_std = prior_std
        self.prior_mean = prior_mean
        self.regularizer_scale = regularizer_scale

    # ==================== abstract methods ==================== #
    @property
    def output_shape(self):
        return [tuple(i[:-1]) + (self.num_units,) for i in self.input_shape]

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)
        outputs = []
        # ====== check inputs (only 2D) ====== #
        X = self.get_input(training, **kwargs)
        # ====== calculate hidden activation ====== #
        outputs = []
        self.mean = []
        self.logsigma = []
        for x, shape in zip(X, self.input_shape):
            mean = self.nonlinearity(T.dot(X, self.W_mean) + self.b_mean)
            logsigma = self.nonlinearity(T.dot(X, self.W_logsigma) + self.b_logsigma)
            self.mean.append(mean)
            self.logsigma.append(logsigma)
            if training: # training mode (always return hidden)
                if config.backend() == 'theano':
                    eps = self._rng.normal((T.shape(x)[0], self.num_units),
                        mean=0., std=1.)
                else:
                    # if you don't specify first dimension, you will
                    # mess up here. Anyway, no cleaner way for doing this
                    eps = self._rng.normal((shape[0], self.num_units),
                        mean=0., std=1.)
                outputs.append(mean + T.exp(logsigma) * eps)
            else: # prediction mode only return mean
                outputs.append(mean)
            self._last_mean_sigma.append((mean, logsigma))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_regularization(self):
        regu = 0.
        for mean, logsigma in zip(self.mean, self.logsigma):
            regu = regu + T.mean(T.kl_gaussian(
                                 mean, logsigma * 2,
                                 self.prior_mean, T.log(self.prior_std) * 2))
        return regu * self.regularizer_scale + \
        super(VariationalDense, self).get_regularization()


class Embedding(OdinFunction):

    """ Turn positive integers (indexes) into dense vectors of fixed size.
    e.g. given W = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] => we have:
    x = [1, 3, 2] = [[0.1, 0.2], [0.5, 0.6], [0.3, 0.4]]

    Parameters
    ----------
    incoming : a :class:`OdinFunction`, Lasagne :class:`Layer` instance,
               keras :class:`Models` instance, variable, placeholder or
               shape tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings (or the vocabulary size).
        The last embedding will have index input_size - 1.

    output_size : int
        The size of each embedding, the number of dimension each integer index
        will be transformed to.

    W : shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the embedding matrix.
        This should be a matrix with shape ``(input_size, output_size)``.
        See :func:`odin.OdinFunction.create_params` for more information.

    dropout : float(0.0 - 1.0), or variable
        Fraction of the embeddings W to drop. The same drop mask is used for
        each row (vocabulary index) of the W matrix, only enable dropout while
        training and dropout != None.

    Example
    -------
    >>> import number as np
    >>> import odin
    >>> from odin import tensor as T
    >>>
    >>> np.random.seed(13)
    >>> X = np.random.randint(0, 1000, size=(3, 10)).astype('int32')
    >>>
    >>> v = T.placeholder(shape=(None, 10), dtype='int32')
    >>> f = odin.nnet.Embedding(v, input_size=1000, output_size=3,
    ...                         dropout=None, seed=13)
    >>> print(f.output_shape)
    ... # [(None, 10, 3)]
    >>> f = T.function(inputs=f.input_var, outputs=f(True)[0]) # enable dropout

    Note
    ----
    This layer require an incoming is has dtype is integer
    """

    def __init__(self, incoming, input_size, output_size,
        W=T.np_uniform, dropout=None, rescale=True, seed=None, **kwargs):
        super(Embedding, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.create_params(W, (input_size, output_size), name="W",
            regularizable=True, trainable=True)

        self.rescale = rescale
        if dropout is not None and not T.is_variable(dropout):
            dropout = T.variable(dropout, name=self.name + '_dropout')
        self.dropout = dropout
        if dropout is not None:
            self._rng = T.rng(seed)
        else:
            self._rng = None

    @property
    def output_shape(self):
        outshape = []
        for input_shape in self.input_shape:
            outshape.append(input_shape + (self.output_size,))
        return outshape

    def __call__(self, training=False, **kwargs):
        inputs = self.get_input(training, **kwargs)

        # ====== create dropout mask ====== #
        B = None
        if self.dropout is not None:
            retain_p = 1. - self.dropout
            if training:
                B = self._rng.binomial(shape=(self.input_size,), p=retain_p)
            elif self.rescale:
                B = T.ones(shape=(self.input_size,)) / retain_p

        W = self.W
        if B is not None:
            W = W * T.expand_dims(B)

        # ====== embedding ====== #
        outputs = []
        for i in inputs:
            outputs.append(T.gather(W, i))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
