from __future__ import print_function, division, absolute_import

import numpy as np
from six.moves import zip

from .. import tensor as T
from ..base import OdinFunction


def _process_noise_dim(input_shape, dims):
    '''(None, 10, 10) with noise_dims=2
    => (None, 10, 1)
    '''
    # ====== get noise shape ====== #
    if dims is None:
        noise_shape = [None] * len(input_shape)
    else:
        noise_shape = []
        for shape in input_shape:
            noise_shape.append(
                tuple([1 if i in dims else j
                       for i, j in enumerate(shape)])
            )
    return noise_shape


class Dropout(OdinFunction):

    """Dropout functions

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    consitent : bool (default:False)
        in case this function receive multiple inputs, if consistent is True,
        it uses the same dropout mask for all inputs.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, p=0.5,
                 rescale=True,
                 noise_dims=None,
                 consistent=False,
                 seed=None, **kwargs):
        super(Dropout, self).__init__(
            incoming, unsupervised=False, **kwargs)
        self._rng = T.rng(seed=seed)
        if p is not None and not T.is_variable(p):
            p = T.variable(p, name=self.name + '_p')
        self.p = p
        self.rescale = rescale
        if noise_dims is not None and not isinstance(noise_dims, (tuple, list)):
            noise_dims = [noise_dims]
        self.noise_dims = noise_dims
        self.consistent = consistent

    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training=False):
        # return deterministic value
        inputs = self.get_inputs(training)
        if not training or self.p is None:
            outputs = inputs
        else:
            # ====== get noise shape ====== #
            noise_shape = _process_noise_dim(
                self.input_shape, self.noise_dims)
            # ====== dropout each inputs ====== #
            outputs = []
            if not self.consistent:
                for x, shape in zip(inputs, noise_shape):
                    outputs.append(
                        T.dropout(x, self.p, rescale=self.rescale,
                                  noise_shape=shape, rng=self._rng)
                    )
            else:
                seed = self._rng.randint()
                for x, shape in zip(inputs, noise_shape):
                    outputs.append(
                        T.dropout(x, self.p, rescale=self.rescale,
                                  noise_shape=shape, seed=seed)
                    )

        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class FastDropout(OdinFunction):

    """FastDropout functions

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    consitent : bool (default:False)
        in case this function receive multiple inputs, if consistent is True,
        it uses the same dropout mask for all inputs.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, p=0.5,
                 rescale=True,
                 noise_dims=None,
                 consistent=False,
                 seed=None, **kwargs):
        super(Dropout, self).__init__(
            incoming, unsupervised=False, **kwargs)
        self._rng = T.rng(seed=seed)
        if p is not None and not T.is_variable(p):
            p = T.variable(p, name=self.name + '_p')
        self.p = p
        self.rescale = rescale
        if noise_dims is not None and not isinstance(noise_dims, (tuple, list)):
            noise_dims = [noise_dims]
        self.noise_dims = noise_dims
        self.consistent = consistent

    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training=False):
        # return deterministic value
        inputs = self.get_inputs(training)
        if not training or self.p is None:
            outputs = inputs
        else:
            # ====== get noise shape ====== #
            noise_shape = _process_noise_dim(
                self.input_shape, self.noise_dims)
            # ====== dropout each inputs ====== #
            outputs = []
            if not self.consistent:
                for x, shape in zip(inputs, noise_shape):
                    outputs.append(
                        T.dropout(x, self.p, rescale=self.rescale,
                                  noise_shape=shape, rng=self._rng)
                    )
            else:
                seed = self._rng.randint()
                for x, shape in zip(inputs, noise_shape):
                    outputs.append(
                        T.dropout(x, self.p, rescale=self.rescale,
                                  noise_shape=shape, seed=seed)
                    )

        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs


class Noise(OdinFunction):

    """Gaussian noise layer.

    Add zero-mean Gaussian noise of given standard deviation to the input [1]_.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Standard deviation of added Gaussian noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    uniform : bool (default: False)
        whether using uniform noise or the default Gaussian noise
    consitent : bool (default:False)
        in case this function receive multiple inputs, if consistent is True,
        it uses the same noise mask for all inputs.

    Notes
    -----
    The Gaussian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.

    References
    ----------
    .. [1] K.-C. Jim, C. Giles, and B. Horne (1996):
           An analysis of noise in recurrent neural networks: convergence and
           generalization.
           IEEE Transactions on Neural Networks, 7(6):1424-1438.
    """

    def __init__(self, incoming, sigma=0.1, noise_dims=None,
                 uniform=False, consistent=False, seed=None, **kwargs):
        super(Noise, self).__init__(incoming, unsupervised=False, **kwargs)
        self._rng = T.rng(seed=seed)
        if sigma is not None and not T.is_variable(sigma):
            sigma = T.variable(sigma, name=self.name + '_p')
        self.sigma = sigma
        if noise_dims is not None and not isinstance(noise_dims, (tuple, list)):
            noise_dims = [noise_dims]
        self.noise_dims = noise_dims
        self.uniform = uniform
        self.consistent = consistent

    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training=False):
        inputs = self.get_inputs(training)
        if not training or self.sigma is None:
            outputs = inputs
        else:
            # ====== get noise shape ====== #
            noise_shape = _process_noise_dim(
                self.input_shape, self.noise_dims)
            # ====== create random function ====== #
            if not self.consistent:
                if not self.uniform:
                    randfunc = lambda x, shape: self._rng.normal(
                        shape=shape, mean=0.0,
                        std=self.sigma, dtype=x.dtype)
                else:
                    randfunc = lambda x, shape: self._rng.uniform(
                        shape=shape, low=-self.sigma,
                        high=self.sigma, dtype=x.dtype)
            else:
                seed = self._rng.randint()
                if not self.uniform:
                    randfunc = lambda x, shape: T.random_normal(
                        shape=shape, mean=0.0,
                        std=self.sigma, dtype=x.dtype, seed=seed)
                else:
                    randfunc = lambda x, shape: T.random_uniform(
                        shape=shape, low=-self.sigma,
                        high=self.sigma, dtype=x.dtype, seed=seed)

            # ====== dropout each inputs ====== #
            outputs = []
            for x, shape in zip(inputs, noise_shape):
                broadcastable = None
                if shape is None:
                    shape = T.shape(x)
                else:
                    broadcastable = [i for i, j in enumerate(shape) if j == 1]
                noise = randfunc(x, shape)
                if self.uniform:
                    # no idea why uniform does not give any broadcastable
                    # dimensions
                    if broadcastable is not None and len(broadcastable) > 0:
                        noise = T.addbroadcast(noise, *broadcastable)
                outputs.append(x + noise)
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, inputs, outputs)
        return outputs
