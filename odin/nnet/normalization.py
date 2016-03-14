# -*- coding: utf-8 -*-
# ===========================================================================
# This module is created based on the code from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division

from six.moves import zip_longest, zip, range
import numpy as np

from .. import config
from .. import tensor as T
from ..base import OdinFunction

__all__ = [
    "BatchNormalization",
]


class BatchNormalization(OdinFunction):

    """ Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=T.np_constant, gamma =lambda x: T.np_constant(x, 1.),
                 mean=T.np_constant, inv_std=lambda x: T.np_constant(x, 1.),
                 nonlinearity=T.linear,
                 **kwargs):
        super(BatchNormalization, self).__init__(
            incoming, unsupervised=False, **kwargs)

        input_shape = self.input_shape[0]
        for i in self.input_shape:
            if i[1:] != input_shape[1:]:
                self.raise_arguments('All dimensions from the second dimensions'
                                     ' of all inputs must be similar, but '
                                     '{} != {}'.format(i, input_shape))
        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            self.raise_arguments("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.create_params(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.create_params(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.create_params(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.create_params(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

        if nonlinearity is None or not hasattr(nonlinearity, '__call__'):
            nonlinearity = T.linear
        self.nonlinearity = nonlinearity

    # ==================== abstract methods ==================== #
    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training, **kwargs):
        batch_norm_use_averages = kwargs.get('batch_norm_use_averages', not training)
        batch_norm_update_averages = kwargs.get('batch_norm_update_averages', training)
        inputs = self.get_input(training, **kwargs)
        outputs = []
        for input in inputs:
            input_mean = T.mean(input, self.axes)
            input_inv_std = 1. / T.sqrt(T.var(input, self.axes) + self.epsilon)

            # Decide whether to use the stored averages or mini-batch statistics
            if batch_norm_use_averages:
                mean = self.mean
                inv_std = self.inv_std
            else:
                mean = input_mean
                inv_std = input_inv_std

            if batch_norm_update_averages:
                if config.backend() == 'theano':
                    # this trick really fast and efficency so I want to keep it
                    import theano
                    # Trick: To update the stored statistics, we create memory-aliased
                    # clones of the stored statistics:
                    running_mean = theano.clone(self.mean, share_inputs=False)
                    running_inv_std = theano.clone(self.inv_std, share_inputs=False)
                    # set a default update for them:
                    running_mean.default_update = ((1 - self.alpha) * running_mean +
                                                   self.alpha * input_mean)
                    running_inv_std.default_update = ((1 - self.alpha) *
                                                      running_inv_std +
                                                      self.alpha * input_inv_std)
                    # and make sure they end up in the graph without participating in
                    # the computation (this way their default_update will be collected
                    # and applied, but the computation will be optimized away):
                    mean += 0 * running_mean
                    inv_std += 0 * running_inv_std
                elif config.backend() == 'tensorflow':
                    T.add_global_updates(self.mean, ((1 - self.alpha) * self.mean +
                                                   self.alpha * input_mean))
                    T.add_global_updates(self.inv_std, ((1 - self.alpha) *
                                                      self.inv_std +
                                                      self.alpha * input_inv_std))

            # prepare dimshuffle pattern inserting broadcastable axes as needed
            param_axes = iter(range(T.ndim(input) - len(self.axes)))
            pattern = ['x' if input_axis in self.axes
                       else next(param_axes)
                       for input_axis in range(T.ndim(input))]

            # apply dimshuffle pattern to all parameters
            beta = 0. if self.beta is None else T.dimshuffle(self.beta, pattern)
            gamma = 1. if self.gamma is None else T.dimshuffle(self.gamma, pattern)
            mean = T.dimshuffle(mean, pattern)
            inv_std = T.dimshuffle(inv_std, pattern)

            # normalize
            normalized = (input - mean) * (gamma * inv_std) + beta
            outputs.append(self.nonlinearity(normalized))
        # ====== foot_print ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
