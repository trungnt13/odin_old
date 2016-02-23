from __future__ import print_function, division, absolute_import

import numpy as np
from six.moves import zip

from .. import tensor as T
from .. base import OdinFunction

class AutoEncoderDecoder(OdinFunction):

    def __init__(self, encoder, decoder, **kwargs):
        if not isinstance(encoder, OdinFunction) or \
           not isinstance(decoder, OdinFunction):
           self.raise_arguments('AutoEncoderDecoder only support OdinFunction'
                                'as incoming inputs')

        # only modify root if 2 different graphs
        if encoder not in decoder.get_children():
            for i in decoder.get_roots():
                if encoder not in i.incoming:
                    i.set_incoming(encoder)

        super(AutoEncoderDecoder, self).__init__(
            incoming=decoder, unsupervised=True, strict_batch=False, **kwargs)

        self._encoder = encoder
        self._decoder = decoder

        # this mode return output similar to input
        self._reconstruction_mode = True

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        return self._decoder.output_shape

    def __call__(self, training=False, **kwargs):
        if 'output_hidden' in kwargs:
            self._reconstruction_mode = not kwargs['output_hidden']
        return self.get_inputs(training)

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        x_reconstructed = self(training=training)
        if not isinstance(x_reconstructed, (tuple, list)):
            x_reconstructed = [x_reconstructed]
        x_original = []
        for i in self._encoder.get_roots():
            x_original += i._last_inputs

        # ====== Objectives is mean of all in-out pair ====== #
        obj = T.castX(0.)
        for i, j in zip(x_reconstructed, x_original):
            if T.ndim(i) != T.ndim(j):
                j = T.reshape(j, T.shape(i))

            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            o = objective(i, j)
            if T.ndim(o) > 0:
                o = T.mean(T.sum(o, axis=-1))
            obj = obj + o

        # ====== Create the optimizer ====== #
        if optimizer is None:
            return obj, None
        params = self.get_params(globals=globals, trainable=True)
        if globals:
            grad = T.gradients(obj, params)
        else:
            grad = T.gradients(obj, params, consider_constant=x_original)
        opt = optimizer(grad, params)
        return obj, opt

    def get_inputs(self, training=True):
        if self._reconstruction_mode:
            return self._decoder(training)
        else:
            self._reconstruction_mode = True
            return self._encoder(training)

    @property
    def input_var(self):
        return self._encoder.input_var

    def get_params(self, globals, trainable=None, regularizable=None):
        ''' Weights must be tied '''
        return self._encoder.get_params(globals, trainable, regularizable)

class AutoEncoder(OdinFunction):

    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    Parameters
    ----------
    incoming : a :class:`OdinFunction`, Lasagne :class:`Layer` instance,
               keras :class:`Models` instance, or a tuple
        The layer feeding into this layer, or the expected input shape.

    num_units : int
        number of hidden units

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases.

    denoising : float(0.0-1.0)
        corruption level of input for denoising autoencoder. If the value is 0.,
        no corruption is applied

    seed : int
        seed for RandomState used for sampling

    """

    def __init__(self, incoming, num_units,
        W=T.np_glorot_uniform, b=T.np_constant,
        denoising=0.5, seed=None, **kwargs):
        super(AutoEncoder, self).__init__(
            incoming, unsupervised=True, strict_batch=False, **kwargs)

        self.num_units = num_units
        self.denoising = denoising
        self._rng = T.rng(seed)

        shape = (np.prod(self.input_shape[0][1:]), num_units)
        self.W = self.create_params(
            W, shape, 'W', regularizable=True, trainable=True)
        if b is None:
            self.hbias = None
            self.vbias = None
        else:
            self.hbias = self.create_params(b, (shape[1],), 'hbias',
                regularizable=False, trainable=True)
            self.vbias = self.create_params(b, (shape[0],), 'vbias',
                regularizable=False, trainable=True)

        # b_prime corresponds to the bias of the visible
        self.b_prime = self.vbias
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        return self.input_shape

    def __call__(self, training=False, **kwargs):
        if 'inputs' in kwargs:
            X = kwargs['inputs']
            if not isinstance(X, (tuple, list)):
                X = [X]
        else:
            X = self.get_inputs(training)
        X = [x if T.ndim(x) <= 2 else T.flatten(x, 2) for x in X]

        outputs = []
        for x, shape in zip(X, self.output_shape):
            hidden_state = T.sigmoid(T.dot(x, self.W) + self.hbias)
            reconstructed = T.sigmoid(
                T.dot(hidden_state, self.W_prime) + self.b_prime)
            if not training:
                reconstructed = T.reshape(reconstructed,
                                        (-1,) + tuple(shape[1:]))
            outputs.append(reconstructed)
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        X = [x if T.ndim(x) <= 2 else T.flatten(x, 2)
            for x in self.get_inputs(training)]
        x_corrupted = [self._get_corrupted_input(x, self.denoising) for x in X]
        x_reconstructed = self(training=training, inputs=x_corrupted)

        obj = T.castX(0.)
        for xr, xc in zip(x_reconstructed, x_corrupted):
            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            o = objective(xr, xc)
            if T.ndim(o) > 0:
                o = T.mean(T.sum(o, axis=-1))
            obj = obj + o

        if optimizer is None:
            return obj, None

        params = self.get_params(globals=globals, trainable=True)
        if globals:
            grad = T.gradients(obj, params)
        else:
            grad = T.gradients(obj, params, consider_constant=X)
        opt = optimizer(grad, params)
        return obj, opt

    # ==================== helper function ==================== #
    def _get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        this will produce an array of 0s and 1s where 1 has a probability of
        1 - ``corruption_level`` and 0 with ``corruption_level``
        """
        return self._rng.binomial(
            shape=input.shape, p=1 - corruption_level) * input
