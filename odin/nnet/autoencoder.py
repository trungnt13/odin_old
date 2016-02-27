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
                i._intermediate_inputs = inputs
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

    """

    def __init__(self, incoming, num_units,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.sigmoid,
                 denoising=0.5, seed=None, **kwargs):
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
            outputs.append(reconstructed)
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
        self._intermediate_inputs = x_corrupted # dirty hack modify inputs
        tmp = self._reconstruction_mode
        x_reconstructed = self.set_reconstruction_mode(True)(training)
        self.set_reconstruction_mode(tmp) # reset the mode

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

class VAE(object):

    '''Variational Auto-encoder Class for image data.
    Using linear/convolutional transformations with ReLU activations, this
    class peforms an encoding and then decoding step to form a full generative
    model for image data. Images can thus be encoded to a latent representation-space
    or decoded/generated from latent vectors.
    See  Kingma, Diederik and Welling, Max; "Auto-Encoding Variational Bayes"
    (2013)
    Given a set of input images train an artificial neural network, resampling
    at the latent stage from an approximate posterior multivariate gaussian
    distribution with unit covariance with mean and variance trained by the
    encoding step.
    Attributes
    ----------
    encode_layers : List[int]
        List of layer sizes for hidden linear encoding layers of the model.
        Only taken into account when mode='linear'.
    decode_layers : List[int]
        List of layer sizes for hidden linear decoding layers of the model.
        Only taken into account when mode='linear'.
    latent_width : int
        Dimension of latent encoding space.
    img_width : int
        Width of the desired image representation.
    img_height : int
        Height of the desired image representation.
    color_channels : int
        Number of color channels in the input images.
    kl_ratio : float
            Multiplicative factor on and KL Divergence term.
    flag_gpu : bool
        Flag to toggle GPU training functionality.
    mode: str
        Mode to set the encoder and decoder architectures. Can be either
        'convolution' or 'linear'.
    adam_alpha : float
        Alpha parameter for the adam optimizer.
    adam_beta1 : float
        Beta1 parameter for the adam optimizer.
    rectifier : str
        Rectifier option for the output of the decoder. Can be either
        'clipped_relu' or 'sigmoid'.
    model : chainer.Chain
        Chain of chainer model links for encoding and decoding.
    opt : chainer.Optimizer
        Chiner optimizer used to do backpropagation. Set to Adam.
    '''

    def __init__(self, img_width=64, img_height=64, color_channels=3,
                 encode_layers=[1000, 600, 300],
                 decode_layers=[300, 800, 1000],
                 latent_width=100, kl_ratio=1.0, flag_gpu=True,
                 mode='convolution', adam_alpha=0.001, adam_beta1=0.9,
                 rectifier='clipped_relu'):

        self.img_width = img_width
        self.img_height = img_height
        self.color_channels = color_channels
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.latent_width = latent_width
        self.kl_ratio = kl_ratio
        self.flag_gpu = flag_gpu
        self.mode = mode
        self.adam_alpha = adam_alpha
        self.adam_beta1 = adam_beta1
        self.rectifier = rectifier
        if self.mode == 'convolution':
            self._check_dims()

        self.model = EncDec(
            img_width=self.img_width,
            img_height=self.img_height,
            color_channels=self.color_channels,
            encode_layers=self.encode_layers,
            decode_layers=self.decode_layers,
            latent_width=self.latent_width,
            mode=self.mode,
            flag_gpu=self.flag_gpu,
            rectifier=self.rectifier
        )
        if self.flag_gpu:
            self.model = self.model.to_gpu()

        self.opt = O.Adam(alpha=self.adam_alpha, beta1=self.adam_beta1)

    def _check_dims(self):
        h, w = calc_fc_size(self.img_height, self.img_width)[1:]
        h, w = calc_im_size(h, w)

        assert (h == self.img_height) and (w == self.img_width),\
            "To use convolution, please resize images " + \
            "to nearest target height, width = %d, %d" % (h, w)

    def _save_meta(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        d = self.__dict__.copy()
        d.pop('model')
        d.pop('opt')
        # d.pop('xp')
        meta = json.dumps(d)
        with open(filepath + '.json', 'wb') as f:
            f.write(meta)

    def transform(self, data, test=False):
        '''Transform image data to latent space.
        Parameters
        ----------
        data : array-like shape (n_images, image_width, image_height,
                                   n_colors)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.
        Returns
        -------
        latent_vec : array-like shape (n_images, latent_width)
        '''
        #make sure that data has the right shape.
        if not type(data) == Variable:
            if len(data.shape) < 4:
                data = data[np.newaxis]
            if len(data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.shape))
            if data.shape[1] != self.color_channels:
                if data.shape[-1] == self.color_channels:
                    data = data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))
            data = Variable(data)
        else:
            if len(data.data.shape) < 4:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 4:
                raise TypeError("Invalid dimensions for image data. Dim = %s.\
                     Must be 4d array." % str(data.data.shape))
            if data.data.shape[1] != self.color_channels:
                if data.data.shape[-1] == self.color_channels:
                    data.data = data.data.transpose(0, 3, 1, 2)
                else:
                    raise TypeError("Invalid dimensions for image data. Dim = %s"
                                    % str(data.shape))

        # Actual transformation.
        if self.flag_gpu:
            data.to_gpu()
        z = self.model.encode(data, test=test)[0]

        z.to_cpu()

        return z.data

    def inverse_transform(self, data, test=False):
        '''Transform latent vectors into images.
        Parameters
        ----------
        data : array-like shape (n_images, latent_width)
            Input numpy array of images.
        test [optional] : bool
            Controls the test boolean for batch normalization.
        Returns
        -------
        images : array-like shape (n_images, image_width, image_height,
                                   n_colors)
        '''
        if not type(data) == Variable:
            if len(data.shape) < 2:
                data = data[np.newaxis]
            if len(data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.shape))
            data = Variable(data)

        else:
            if len(data.data.shape) < 2:
                data.data = data.data[np.newaxis]
            if len(data.data.shape) != 2:
                raise TypeError("Invalid dimensions for latent data. Dim = %s.\
                     Must be a 2d array." % str(data.data.shape))
        assert data.data.shape[-1] == self.latent_width,\
            "Latent shape %d != %d" % (data.data.shape[-1], self.latent_width)

        if self.flag_gpu:
            data.to_gpu()
        out = self.model.decode(data, test=test)

        out.to_cpu()

        if self.mode == 'linear':
            final = out.data
        else:
            final = out.data.transpose(0, 2, 3, 1)

        return final

    def load_images(self, filepaths):
        '''Load in image files from list of paths.
        Parameters
        ----------
        filepaths : List[str]
            List of file paths of images to be loaded.
        Returns
        -------
        images : array-like shape (n_images, n_colors, image_width, image_height)
            Images normalized to have pixel data range [0,1].
        '''
        def read(fname):
            im = Image.open(fname)
            im = np.float32(im)
            return im / 255.
        x_all = np.array([read(fname) for fname in tqdm.tqdm(filepaths)])
        x_all = x_all.astype('float32')
        if self.mode == 'convolution':
            x_all = x_all.transpose(0, 3, 1, 2)
        print("Image Files Loaded!")
        return x_all

    def fit(
        self,
        img_data,
        save_freq=-1,
        pic_freq=-1,

        n_epochs=100,
        batch_size=50,
        weight_decay=True,
        model_path='./VAE_training_model/',
        img_path='./VAE_training_images/',
        img_out_width=10
    ):
        '''Fit the VAE model to the image data.
        Parameters
        ----------
        img_data : array-like shape (n_images, n_colors, image_width, image_height)
            Images used to fit VAE model.
        save_freq [optional] : int
            Sets the number of epochs to wait before saving the model and optimizer states.
            Also saves image files of randomly generated images using those states in a
            separate directory. Does not save if negative valued.
        pic_freq [optional] : int
            Sets the number of batches to wait before displaying a picture or randomly
            generated images using the current model state.
            Does not display if negative valued.
        n_epochs [optional] : int
            Gives the number of training epochs to run through for the fitting
            process.
        batch_size [optional] : int
            The size of the batch to use when training. Note: generally larger
            batch sizes will result in fater epoch iteration, but at the const
            of lower granulatity when updating the layer weights.
        weight_decay [optional] : bool
            Flag that controls adding weight decay hooks to the optimizer.
        model_path [optional] : str
            Directory where the model and optimizer state files will be saved.
        img_path [optional] : str
            Directory where the end of epoch training image files will be saved.
        img_out_width : int
            Controls the number of randomly genreated images per row in the output
            saved imags.
        '''
        width = img_out_width
        self.opt.setup(self.model)

        if weight_decay:
            self.opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

        n_data = img_data.shape[0]

        batch_iter = list(range(0, n_data, batch_size))
        n_batches = len(batch_iter)
        save_counter = 0
        for epoch in range(1, n_epochs + 1):
            print('epoch: %i' % epoch)
            t1 = time.time()
            indexes = np.random.permutation(n_data)
            last_loss_kl = 0.
            last_loss_rec = 0.
            count = 0
            for i in tqdm.tqdm(batch_iter):

                x_batch = Variable(img_data[indexes[i: i + batch_size]])

                if self.flag_gpu:
                    x_batch.to_gpu()

                out, kl_loss, rec_loss = self.model.forward(x_batch)
                total_loss = rec_loss + kl_loss * self.kl_ratio

                self.opt.zero_grads()
                total_loss.backward()
                self.opt.update()

                last_loss_kl += kl_loss.data
                last_loss_rec += rec_loss.data
                plot_pics = Variable(img_data[indexes[:width]])
                count += 1
                if pic_freq > 0:
                    assert type(pic_freq) == int, "pic_freq must be an integer."
                    if count % pic_freq == 0:
                        fig = self._plot_img(
                            plot_pics,
                            img_path=img_path,
                            epoch=epoch
                        )
                        display(fig)

            if save_freq > 0:
                save_counter += 1
                assert type(save_freq) == int, "save_freq must be an integer."
                if epoch % save_freq == 0:
                    name = "vae_epoch%s" % str(epoch)
                    if save_counter == 1:
                        save_meta = True
                    else:
                        save_meta = False
                    self.save(model_path, name, save_meta=save_meta)
                    fig = self._plot_img(
                        plot_pics,
                        img_path=img_path,
                        epoch=epoch,
                        batch=n_batches,
                        save=True
                    )

            msg = "rec_loss = {0} , kl_loss = {1}"
            print(msg.format(last_loss_rec / n_batches, last_loss_kl / n_batches))
            t_diff = time.time() - t1
            print("time: %f\n\n" % t_diff)

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
                 seed=None,
                 **kwargs):
        super(VariationalEncoderDecoder, self).__init__(
            incoming=encoder, strict_batch=False, **kwargs)
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        self.regularizer_scale = regularizer_scale
        self._rng = T.rng(seed=seed)
        self.nonlinearity = nonlinearity
        # ====== Set encoder decoder ====== #
        self.encoder = encoder
        self.decoder = decoder
        # ====== Validate input_shapes (output_shape from encoder) ====== #
        self.input_dim = self._validate_nD_input(2)[1]
        # ====== Validate output_shapes (input_shape from decoder) ====== #
        if self.encoder in self.decoder.get_children():
            self.raise_arguments('Encoder and Decoder must NOT connected, '
                                'because we will inject the VariationalDense '
                                'between encoder-decoder.')
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
        # if any roots contain this, it means the function connected
        for i in self.decoder.get_roots():
            if self not in i.incoming:
                self._decoder_connected = True
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
        # this list store last statistic (mean, logsigma) used to calculate
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
        outshape = []
        if self._reconstruction_mode:
            for i in self.get_roots():
                outshape += i.input_shape
        else:
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
        # ====== Process each input ====== #
        outputs = []
        self._last_mean_sigma = []
        for x in X:
            mean, logsigma = self.get_mean_logsigma(x)
            if training: # training mode (always return hidden)
                if config.backend() == 'theano':
                    eps = self._rng.normal((x.shape[0], self.num_units),
                        mean=0., std=1.)
                else:
                    eps = self._rng.normal((T.shape(x)[0], self.num_units),
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
                i._intermediate_inputs = outputs
            outputs = self.decoder(training)
            if not training:
                tmp = []
                for i, j in zip(outputs, self.output_shape):
                    if T.ndim(i) != len(j):
                        tmp.append(
                            T.reshape(i, (-1,) + tuple(j[1:]))
                        )
                outputs = tmp
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
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
            o = o + T.kl_gaussian(mean, logsigma,
                                  regularizer_scale=self.regularizer_scale,
                                  prior_mean=self.prior_mean,
                                  prior_logsigma=self.prior_logsigma)
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
