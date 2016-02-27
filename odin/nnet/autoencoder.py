# VB calibration to improve the interface between phone recognizer and i-vector extractor
from __future__ import print_function, division, absolute_import

import numpy as np
from six.moves import zip

from .. import tensor as T
from ..base import OdinUnsupervisedFunction, OdinFunction
from .. import config

class AutoEncoderDecoder(OdinUnsupervisedFunction):

    def __init__(self, encoder, decoder,
                 reconstruction=False, **kwargs):
        if not isinstance(encoder, OdinFunction) or \
           not isinstance(decoder, OdinFunction):
           self.raise_arguments('AutoEncoderDecoder only support OdinFunction'
                                'as incoming inputs')

        # this mode is enable if encoder and decoder already connected
        self._connected_encoder_decoder = False
        if encoder in decoder.get_children():
            self.log('Found connected graph of encoder-decoder!', 30)
            self._connected_encoder_decoder = True
        else: # check input-output shape of encoder-decoder
            decoder_out = []
            for i in decoder.get_roots(): # check input_shape from root
                decoder_out += i.input_shape
            for i, j in zip(encoder.output_shape, decoder_out):
                if i[1:] != j[1:]:
                    self.raise_arguments(
                        'Encoder output_shape must match decoder'
                        ' input_shape, but %s != %s' % (i[1:], j[1:]))

        super(AutoEncoderDecoder, self).__init__(
            incoming=encoder, strict_batch=False, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        if self._reconstruction_mode:
            return self.decoder.output_shape
        else:
            return self.encoder.output_shape

    def __call__(self, training=False):
        # ====== Only get hidden states ====== #
        if not self._reconstruction_mode:
            inputs = self.get_inputs(training)
            outputs = inputs
        # ====== Get reconstructed inputs from disconnected graph ====== #
        elif not self._connected_encoder_decoder:
            inputs = self.get_inputs(training)
            # intercept the inputs of decoder
            for i in self.decoder.get_roots():
                i.set_intermediate_inputs(inputs)
            outputs = self.decoder(training)
        # ====== Get reconstructed inputs from connected graph ====== #
        else:
            outputs = self.decoder(training)
            inputs = []
        self._log_footprint(training, inputs, outputs)
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        tmp = self._reconstruction_mode
        x_reconstructed = self.set_reconstruction_mode(True)(training=training)
        self.set_reconstruction_mode(tmp) # reset the reconstruction mode
        if not isinstance(x_reconstructed, (tuple, list)):
            x_reconstructed = [x_reconstructed]
        x_original = []
        for i in self.encoder.get_roots():
            x_original += i._last_inputs
        # ====== Objectives is mean of all in-out pair ====== #
        obj = T.castX(0.)
        for i, j in zip(x_reconstructed, x_original):
            # reshape if reconstructed and original in different shape
            if T.ndim(i) != T.ndim(j):
                j = T.reshape(j, T.shape(i))

            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            o = objective(i, j)
            if T.ndim(o) > 0:
                o = T.mean(T.sum(o, axis=-1))
            obj = obj + o
        obj = obj / len(x_reconstructed)

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

    def get_params(self, globals, trainable=None, regularizable=None):
        ''' Weights must be tied '''
        encoder_params = self.encoder.get_params(
            globals, trainable, regularizable)
        decoder_params = self.decoder.get_params(
            globals, trainable, regularizable)
        params = encoder_params + decoder_params
        # we only consider the variable, not the TensorType (e.g. W.T)
        # Have overlap between encoder and decoder params if globals is turned
        # on, hence we use ordered set
        return T.np_ordered_set([i for i in params if T.is_variable(i)]).tolist()

class AutoEncoder(OdinUnsupervisedFunction):

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

    contractive : boolean (default: false)
        The contractive autoencoder tries to reconstruct the input with an
        additional constraint on the latent space. With the objective of
        obtaining a robust representation of the input space, we
        regularize the L2 norm(Froebenius) of the jacobian of the hidden
        representation with respect to the input. Please refer to Rifai et
        al.,2011 for more details.
        Note: this mode significantly reduce the trianing speed

    """

    def __init__(self, incoming, num_units,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.sigmoid,
                 denoising=0.5,
                 contractive=False, contraction_level=.1,
                 seed=None, **kwargs):
        super(AutoEncoder, self).__init__(
            incoming, strict_batch=False, **kwargs)

        self.num_units = num_units
        self.denoising = denoising
        self._rng = T.rng(seed)

        shape = (self._validate_nD_input(2)[1], num_units)
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
        self.nonlinearity = nonlinearity

        self.contractive = contractive
        self.contraction_level = contraction_level

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        if self._reconstruction_mode:
            return self.input_shape
        return [(i[0], self.num_units) for i in self.input_shape]

    def __call__(self, training=False):
        if training and self._reconstruction_mode:
            self.log('In training mode, the autoencoder does not reshape '
                'to match input_shape when enabling reconstruction mode.', 30)
        # ====== inputs ====== #
        X = [x if T.ndim(x) <= 2 else T.flatten(x, 2)
            for x in self.get_inputs(training)]
        outputs = []
        J = T.castX(0.) # jacobian regularization
        for x, shape in zip(X, self.output_shape):
            hidden_state = self.nonlinearity(T.dot(x, self.W) + self.hbias)
            # output reconstructed data
            if self._reconstruction_mode:
                reconstructed = self.nonlinearity(
                    T.dot(hidden_state, self.W_prime) + self.b_prime)
                # reshape if shape mismatch (but only when not training)
                if not training and T.ndim(reconstructed) != len(shape):
                    reconstructed = T.reshape(
                        reconstructed, (-1,) + tuple(shape[1:]))
            # only output hidden activations
            else:
                reconstructed = hidden_state
            # calculate jacobian
            if self.contractive:
                J = J + T.jacobian_regularize(hidden_state, self.W)
            outputs.append(reconstructed)
        self._jacobian_regularization = J / len(X) * self.contraction_level
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, X, outputs)
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        X = [x if T.ndim(x) <= 2 else T.flatten(x, 2)
            for x in self.get_inputs(training)]
        x_corrupted = [self._get_corrupted_input(x, self.denoising) for x in X]
        self.set_intermediate_inputs(x_corrupted) # dirty hack modify inputs

        tmp = self._reconstruction_mode
        x_reconstructed = self.set_reconstruction_mode(True)(training)
        self.set_reconstruction_mode(tmp) # reset the mode

        # ====== Calculate the objectives ====== #
        obj = T.castX(0.)
        for xr, xc in zip(x_reconstructed, x_corrupted):
            # note : we sum over the size of a datapoint; if we are using
            #        minibatches, L will be a vector, with one entry per
            #        example in minibatch
            o = objective(xr, xc)
            if T.ndim(o) > 0:
                o = T.mean(T.sum(o, axis=-1))
            obj = obj + o

        if self.contractive:
            obj = obj + self._jacobian_regularization

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
    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def _get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        this will produce an array of 0s and 1s where 1 has a probability of
        1 - ``corruption_level`` and 0 with ``corruption_level``
        """
        return self._rng.binomial(
            shape=input.shape, p=1 - corruption_level) * input

class VariationalEncoderDecoder(OdinUnsupervisedFunction):

    """ Hidden layer for Variational Autoencoding Bayes method [1].
    This layer projects the input twice to calculate the mean and variance
    of a Gaussian distribution.
    During training, the output is sampled from that distribution as
    mean + random_noise * variance, during testing the output is the mean,
    i.e the expected value of the encoded distribution.

    Parameters:
    -----------
    batch_size: Both Keras backends need the batch_size to be defined before
        hand for sampling random numbers. Make sure your batch size is kept
        fixed during training. You can use any batch size for testing.
    regularizer_scale: By default the regularization is already proberly
        scaled if you use binary or categorical crossentropy cost functions.
        In most cases this regularizers should be kept fixed at one.

    """

    def __init__(self, encoder, decoder,
                 W=T.np_glorot_uniform, b=T.np_constant,
                 nonlinearity=T.tanh,
                 regularizer_scale=1.,
                 prior_mean=0., prior_logsigma=1.,
                 seed=None, batch_size=None,
                 **kwargs):
        strict_batch = False
        self.batch_size = batch_size
        super(VariationalEncoderDecoder, self).__init__(
            incoming=encoder, strict_batch=strict_batch, **kwargs)
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        self.regularizer_scale = regularizer_scale
        self._rng = T.rng(seed=seed)
        self.nonlinearity = nonlinearity
        # ====== Set encoder decoder ====== #
        self.encoder = encoder
        self.decoder = decoder
        # ====== Validate encoder, decoder ====== #
        if self.encoder in self.decoder.get_children():
            self.raise_arguments('Encoder and Decoder must NOT connected, '
                                'because we will inject the VariationalDense '
                                'between encoder-decoder.')
        # if any roots contain this, it means the function connected
        for i in self.decoder.get_roots():
            if self in i.incoming:
                self.raise_arguments('Decoder and VariationalEncoderDecoder '
                                    'must not connected.')
        # ====== Validate input_shapes (output_shape from encoder) ====== #
        self.input_dim = self._validate_nD_input(2)[1]
        # ====== Validate output_shapes (input_shape from decoder) ====== #
        shapes = []
        for i in self.decoder.get_roots():
            shapes += i.input_shape
        i = shapes[0][1:]
        for j in shapes:
            if i != j[1:]:
                self.raise_arguments(
                    'All the feautre dimensions of decoder input_shape '
                    'must be similar, but %s != %s' % (str(i), str(j)))
        self.num_units = int(np.prod(i))
        # ====== create params ====== #
        self.W_mean = self.create_params(
            W, (self.input_dim, self.num_units), 'W_mean',
            regularizable=True, trainable=True)
        self.b_mean = self.create_params(
            b, (self.num_units,), 'b_mean',
            regularizable=False, trainable=True)
        self.W_logsigma = self.create_params(
            W, (self.input_dim, self.num_units), 'W_logsigma',
            regularizable=True, trainable=True)
        self.b_logsigma = self.create_params(
            b, (self.num_units,), 'b_logsigma',
            regularizable=False, trainable=True)
        # this list store last statistics (mean, logsigma) used to calculate
        # the output of this functions, so we can caluclate KL_Gaussian in
        # the optimization methods
        self._last_mean_sigma = []

    # ==================== Helper methods ==================== #
    def get_mean_logsigma(self, X):
        mean = self.nonlinearity(T.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.nonlinearity(T.dot(X, self.W_logsigma) + self.b_logsigma)
        return mean, logsigma

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        if self._reconstruction_mode:
            return self.decoder.output_shape
        outshape = []
        for i in self.input_shape:
            outshape.append((i[0], self.num_units))
        return outshape

    def __call__(self, training=False):
        '''
        Returns
        -------
        prediction, mean, logsigm : if training=True
        prediction(mean) : if training = False
        '''
        if self._reconstruction_mode and training:
            self.log('Training mode always return hidden activations')
        # ====== check inputs (only 2D) ====== #
        X = [x if T.ndim(x) <= 2 else T.flatten(x, 2)
             for x in self.get_inputs(training)]
        # ====== calculate hidden activation ====== #
        outputs = []
        self._last_mean_sigma = []
        for x in X:
            mean, logsigma = self.get_mean_logsigma(x)
            if training: # training mode (always return hidden)
                if config.backend() == 'theano':
                    eps = self._rng.normal((T.shape(x)[0], self.num_units),
                        mean=0., std=1.)
                else:
                    if self.batch_size is None:
                        self.raise_arguments('tensorflow backend requires to '
                                             'know batch size in advanced.')
                    eps = self._rng.normal((self.batch_size, self.num_units),
                        mean=0., std=1.)
                outputs.append(mean + T.exp(logsigma) * eps)
            else: # prediction mode only return mean
                outputs.append(mean)
            self._last_mean_sigma.append((mean, logsigma))
        # ====== log the footprint for debugging ====== #
        self._log_footprint(training, X, outputs)
        # ====== check if return reconstructed inputs ====== #
        if self._reconstruction_mode:
            # intercept the old decoder roots to get reconstruction from
            # current outputs of this functions
            for i in self.decoder.get_roots():
                i.set_intermediate_inputs(outputs)
            outputs = self.decoder(training)
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        # ====== Get the outputs ====== #
        tmp = self._reconstruction_mode
        outputs = self.set_reconstruction_mode(True)(training)
        self.set_reconstruction_mode(tmp)
        inputs = []
        # make encoder input to be equal shape to decoder output
        for i, j in zip(self.encoder.input_var, self.decoder.output_shape):
            # we can only know ndim not the shape of placeholder
            if T.ndim(i) != len(j):
                i = T.reshape(i, (-1,) + j[1:])
            inputs.append(i)
        # ====== calculate objectives ====== #
        obj = T.castX(0.)
        for x_orig, x_reco, stats in zip(inputs, outputs, self._last_mean_sigma):
            mean = stats[0]
            logsigma = stats[1]
            o = objective(x_reco, x_orig)
            if T.ndim(o) > 0:
                o = T.mean(o)
            o = o + self.regularizer_scale * T.kl_gaussian(mean, logsigma,
                                                self.prior_mean,
                                                self.prior_logsigma)
            obj = obj + o
        obj = obj / len(inputs) # mean of all objectives
        # ====== Create the optimizer ====== #
        if optimizer is None:
            return obj, None
        params = self.get_params(globals=globals, trainable=True)
        if globals:
            grad = T.gradients(obj, params)
        else:
            grad = T.gradients(obj, params, consider_constant=inputs)
        opt = optimizer(grad, params)
        return obj, opt

    def get_params(self, globals, trainable=None, regularizable=None):
        ''' Weights must be tied '''
        params = super(VariationalEncoderDecoder, self).get_params(
            globals, trainable, regularizable)
        decoder_params = []
        if globals: # only add encoder and decoder params if globals is on.
            for i in [self.decoder] + self.decoder.get_children():
                decoder_params += i.get_params(
                    globals=False, trainable=trainable, regularizable=regularizable)
        params = params + decoder_params
        # we only consider the variable, not the TensorType (e.g. W.T)
        # Have overlap between encoder and decoder params if globals is turned
        # on, hence we use ordered set
        return T.np_ordered_set([i for i in params if T.is_variable(i)]).tolist()
