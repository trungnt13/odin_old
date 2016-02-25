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
            self.log('Found disconnected graph of encoder-decoder, set incoming'
                     ' of decoder to encoder!')
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
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.sigmoid,
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
        self.nonlinearity = nonlinearity

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
            hidden_state = self.nonlinearity(T.dot(x, self.W) + self.hbias)
            reconstructed = self.nonlinearity(
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

class VariationalAutoEncoder(OdinFunction):

    def __init__(self, incoming, num_units,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 nonlinearity=T.sigmoid,
                 denoising=0.5, seed=None, **kwargs):
        super(VariationalAutoEncoder, self).__init__(
            incoming, unsupervised=True, strict_batch=False, **kwargs)

        self.n_h = 800
        self.n_z = 20
        self.n_t = 1

        self.gaussian = False

        self.params = Parameters()
        n_x = self.data['n_x']
        n_h = self.n_h
        n_z = self.n_z
        n_t = self.n_t
        scale = hp.init_scale

        if hp.load_model and os.path.isfile(self.filename):
            self.params.load(self.filename)
        else:
            with self.params:
                W1 = shared_normal((n_x, n_h), scale=scale)
                W11 = shared_normal((n_h, n_h), scale=scale)
                W111 = shared_normal((n_h, n_h), scale=scale)
                W2 = shared_normal((n_h, n_z), scale=scale)
                W3 = shared_normal((n_h, n_z), scale=scale)
                W4 = shared_normal((n_h, n_h), scale=scale)
                W44 = shared_normal((n_h, n_h), scale=scale)
                W444 = shared_normal((n_z, n_h), scale=scale)
                W5 = shared_normal((n_h, n_x), scale=scale)
                b1 = shared_zeros((n_h,))
                b11 = shared_zeros((n_h,))
                b111 = shared_zeros((n_h,))
                b2 = shared_zeros((n_z,))
                b3 = shared_zeros((n_z,))
                b4 = shared_zeros((n_h,))
                b44 = shared_zeros((n_h,))
                b444 = shared_zeros((n_h,))
                b5 = shared_zeros((n_x,))

        def encoder(x, p):
            h_encoder = T.tanh(T.dot(x, p.W1) + p.b1)
            h_encoder2 = T.tanh(T.dot(h_encoder, p.W11) + p.b11)
            h_encoder3 = T.tanh(T.dot(h_encoder2, p.W111) + p.b111)

            mu_encoder = T.dot(h_encoder3, p.W2) + p.b2
            log_sigma_encoder = 0.5 * (T.dot(h_encoder3, p.W3) + p.b3)
            log_qpz = -0.5 * T.sum(1 + 2 * log_sigma_encoder - mu_encoder**2 - T.exp(2 * log_sigma_encoder))

            eps = srnd.normal(mu_encoder.shape, dtype=theano.config.floatX)
            z = mu_encoder + eps * T.exp(log_sigma_encoder)
            return z, log_qpz

        def decoder(z, p, x=None):
            h_decoder3 = T.tanh(T.dot(z, p.W444) + p.b444)
            h_decoder2 = T.tanh(T.dot(h_decoder3, p.W44) + p.b44)
            h_decoder = T.tanh(T.dot(h_decoder2, p.W4) + p.b4)

            if self.gaussian:
                pxz = T.tanh(T.dot(h_decoder, p.W5) + p.b5)
            else:
                pxz = T.nnet.sigmoid(T.dot(h_decoder, p.W5) + p.b5)

            if not x is None:
                if self.gaussian:
                    log_sigma_decoder = 0
                    log_pxz = 0.5 * np.log(2 * np.pi) + log_sigma_decoder + 0.5 * T.sum(T.sqr(x - pxz))
                else:
                    log_pxz = T.nnet.binary_crossentropy(pxz, x).sum()
                return pxz, log_pxz
            else:
                return pxz

        x = binomial(self.X)
        z, log_qpz = encoder(x, self.params)
        pxz, log_pxz = decoder(z, self.params, x)
        cost = log_pxz + log_qpz

        s_pxz = decoder(self.Z, self.params)
        a_pxz = T.zeros((self.n_t, s_pxz.shape[0], s_pxz.shape[1]))
        a_pxz = T.set_subtensor(a_pxz[0,:,:], s_pxz)

        self.compile(log_pxz, log_qpz, cost, a_pxz)

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
        with open(filepath+'.json', 'wb') as f:
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
            return im/255.
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
                total_loss = rec_loss + kl_loss*self.kl_ratio

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
            print(msg.format(last_loss_rec/n_batches, last_loss_kl/n_batches))
            t_diff = time.time()-t1
            print("time: %f\n\n" % t_diff)

class VariationalDense(OdinFunction):
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
    def __init__(self, incoming, output_dim, batch_size, input_dim=None,
                 W=T.np_glorot_uniform, b=T.np_constant,
                 regularizer_scale=1.,
                 prior_mean=0., prior_logsigma=1.,
                 seed=None,
                 **kwargs):
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        self.regularizer_scale = regularizer_scale

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.initial_weights = weights
        self.input_dim = input_dim

        self._rng = T.rng(seed=seed)
        super(AutoEncoder, self).__init__(
            incoming, unsupervised=True, strict_batch=False, **kwargs)

    def build(self):
        input_dim = self.input_shape[-1]

        self.W_mean = self.init((input_dim, self.output_dim))
        self.b_mean = K.zeros((self.output_dim,))
        self.W_logsigma = self.init((input_dim, self.output_dim))
        self.b_logsigma = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W_mean, self.b_mean, self.W_logsigma,
                       self.b_logsigma]

        self.regularizers = []
        reg = self.get_variational_regularization(self.get_input())
        self.regularizers.append(reg)

    def get_variational_regularization(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return GaussianKL(mean, logsigma,
                          regularizer_scale=self.regularizer_scale,
                          prior_mean=self.prior_mean,
                          prior_logsigma=self.prior_logsigma)

    def get_mean_logsigma(self, X):
        mean = self.activation(K.dot(X, self.W_mean) + self.b_mean)
        logsigma = self.activation(K.dot(X, self.W_logsigma) + self.b_logsigma)
        return mean, logsigma

    def _get_output(self, X, train=False):
        mean, logsigma = self.get_mean_logsigma(X)
        if train:
            if K._BACKEND == 'theano':
                eps = K.random_normal((X.shape[0], self.output_dim))
            else:
                eps = K.random_normal((self.batch_size, self.output_dim))
            return mean + K.exp(logsigma) * eps
        else:
            return mean

    def get_output(self, train=False):
        X = self.get_input()
        return self._get_output(X, train)

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)