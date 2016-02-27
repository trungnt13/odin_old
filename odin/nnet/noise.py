from __future__ import print_function, division, absolute_import

import numpy as np
from six.moves import zip

from .. import tensor as T
from ..base import OdinFunction

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
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.

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

    @property
    def output_shape(self):
        return self.input_shape

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        return self._deterministic_optimization_procedure(
            objective, optimizer, globals, training)

    def __call__(self, training=False):
        # return deterministic value
        inputs = self.get_inputs(training)
        if not training or self.p is None:
            return inputs
        else:
            # ====== get noise shape ====== #
            if self.noise_dims is None:
                noise_shape = [None] * len(inputs)
            else:
                noise_shape = []
                for shape in self.input_shape:
                    noise_shape.append(
                        tuple([1 if i in self.noise_dims else j
                               for i, j in enumerate(shape)])
                    )
            # ====== dropout each inputs ====== #
            outputs = []
            for x, shape in zip(inputs, noise_shape):
                outputs.append(
                    T.dropout(x, self.p, rescale=self.rescale,
                              noise_shape=shape, rng=self._rng)
                )
            return outputs
