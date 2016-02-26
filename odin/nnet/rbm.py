from __future__ import print_function, division, absolute_import

import numpy as np

from .. import tensor as T
from .. base import OdinFunction

class RBM(OdinFunction):

    """Restricted Boltzmann Machine (RBM)
    RBM constructor. Defines the parameters of the model along with
    basic operations for inferring hidden from visible (and vice-versa),
    as well as for performing CD updates.

    Parameters
    ----------
    incoming : a :class:`OdinFunction`, Lasagne :class:`Layer` instance,
               keras :class:`Models` instance, or a tuple
        The layer feeding into this layer, or the expected input shape.

    num_units : int
        number of hidden units

    gibbs_steps : int
        number of Gibbs steps to do in CD-k/PCD-k

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases.

    persistent : None, int (batch_size), variable, ndarray
        persistent is the initial state of sampling chain which has shape
        (batch_size, input_dim). If persisten is not None, the batch_size must
        be fixed and strict_batch is turnned on.

    seed : int
        seed for RandomState used for sampling

    Call
    ----
    gibbs_steps : int
        optional gibbs step used when sampling (this variable will override the
        default parameter)
    output_hidden : bool (default: False)
        if True, when sampling for prediction, return the hidden samples and
        mean instead


    Return
    ------
    [pre_sigmoid_nvs,
     nv_means,
     nv_samples,
     pre_sigmoid_nhs,
     nh_means,
     nh_samples
    ], updates : for training
    [vis_mean[-1],
     vis_samples[-1]
    ], updates : for prediction (reshaped the same as input shape)

    """

    def __init__(self, incoming, num_units,
                 W=T.np_glorot_uniform,
                 b=T.np_constant,
                 gibbs_steps=15, persistent=None,
                 seed=None, **kwargs):
        # ====== persitent variable ====== #
        if persistent is None:
            persistent_params = None
        elif isinstance(persistent, int):
            persistent_params = np.zeros((persistent, num_units))
        elif T.is_variable(persistent):
            persistent_params = persistent
        elif T.is_ndarray(persistent):
            persistent_params = T.variable(persistent)
        else:
            self.raise_arguments('Unsupport type %s for persistent' %
                persistent.__class__.__name__)

        # ====== super ====== #
        if persistent_params is None:
            strict_batch = False
        else:
            strict_batch = persistent
        super(RBM, self).__init__(
            incoming, unsupervised=True, strict_batch=strict_batch, **kwargs)

        # ====== create persitent variable ====== #
        if persistent_params is not None:
            self.persistent = self.create_params(
                persistent_params, shape=None, name='persistent',
                regularizable=False, trainable=False)
        else:
            self.persistent = None

        # ====== others ====== #
        self.n_visible = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self._rng = T.rng(seed)

        # ====== create variables ====== #
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

        self.gibbs_steps = gibbs_steps

    # ==================== Abstract methods ==================== #
    @property
    def output_shape(self):
        # shape = self.input_shape[0]
        # return (shape[0], np.prod(shape[1:]))
        return self.input_shape[0]

    def __call__(self, training=False, **kwargs):
        ''' The sampling process was optimized using loop (unroll_scan) on
        theano which gives significantly speed up '''
        if 'gibbs_steps' in kwargs:
            k = kwargs['gibbs_steps']
        else:
            k = self.gibbs_steps

        reconstructed = False
        if 'reconstructed' in kwargs:
            reconstructed = kwargs['reconstructed']
        if reconstructed and training:
            self.log('Training mode always return hidden activations')

        X = self.get_inputs(training)[0]
        if T.ndim(X) > 2:
            X = T.flatten(X, 2)
        self._last_inputs = [X] # must update last inputs because we reshape X

        # ====== Training ====== #
        if training:
            # decide how to initialize persistent chain:
            # for CD, we use the newly generate hidden sample
            # for PCD, we initialize from the old state of the chain
            if self.persistent is None:
                # compute positive phase
                pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(X)
                chain_start = ph_sample
            else:
                chain_start = self.persistent
            # perform actual negative phase
            # in order to implement CD-k/PCD-k we need to scan over the
            # function that implements one gibbs step k times.
            # Read Theano tutorial on scan for more information :
            # http://deeplearning.net/software/theano/library/scan.html
            # the scan will return the entire Gibbs chain
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ] = T.loop(
                step_fn=self.gibbs_hvh,
                # the None are place holders, saying that
                # chain_start is the initial state corresponding to the
                # 6th output
                outputs_info=[None, None, None, None, None, chain_start],
                n_steps=k
            )
            outputs = [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ]
        # ====== sampling ====== #
        else:
            persistent_vis_chain = X
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ] = T.loop(
                step_fn=self.gibbs_vhv,
                outputs_info=[None, None, None, None, None, persistent_vis_chain],
                n_steps=k
            )
            if reconstructed:
                outputs = T.reshape(vis_mfs[-1], (-1,) + self.output_shape[1:])
            else:
                outputs = hid_mfs[-1]
        self._log_footprint(training, X, outputs)
        return outputs

    def get_optimization(self, objective=None, optimizer=None,
                         globals=True, training=True):
        """This functions implements one step of CD-k or PCD-k

        Returns
        -------
        cost : a proxy for the cost
        updates : dictionary. The dictionary contains the update rules for
        weights and biases but also an update of the shared variable used to
        store the persistent chain, if one is used.

        """
        if objective is not None:
            self.log("Ignored objective:%s because RBM uses contrastive divergence"
                     " as default" % str(objective), 30)
        if training:
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ] = self(training)
            # determine gradients on RBM parameters
            # note that we only need the sample at the end of the chain
            chain_end = nv_samples[-1]
        else:
            self.raise_arguments('Optimization only can be used while training')
        X = self._last_inputs[0] # get cached inputs

        # for the persistent Contrastive-divergent, the chain does not take
        # into account training samples but the free energy does take into
        # account the training samples
        cost = T.mean(self._free_energy(X)) - T.mean(self._free_energy(chain_end))
        if optimizer is None:
            return cost, None

        params = self.get_params(globals=globals, trainable=True)
        # We must not compute the gradient through the gibbs sampling
        if globals:
            gparams = T.gradients(cost, params, consider_constant=[chain_end])
        else:
            gparams = T.gradients(cost, params, consider_constant=[chain_end, X])
        updates = optimizer(gparams, params)

        # ====== create monitoring cost ====== #
        if training:
            if self.persistent:
                # Note that this works only if persistent is a shared variable
                updates[self.persistent] = nh_samples[-1]
                # pseudo-likelihood is a better proxy for PCD
                monitoring_cost = self._get_pseudo_likelihood_cost(X, updates)
            else:
                # reconstruction cross-entropy is a better proxy for CD
                monitoring_cost = self._get_reconstruction_cost(
                    X, updates, pre_sigmoid_nvs[-1])
        else:
            monitoring_cost = cost

        return monitoring_cost, updates

    # ==================== Energy methods ==================== #
    def _free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # propagate upwards to hidden units
        # Note that we return also the pre-sigmoid activation of the
        # layer. As it will turn out later, due to how Theano deals with
        # optimizations, this symbolic variable will be needed to write
        # down a more stable computational graph (see details in the
        # reconstruction cost function)
        pre_sigmoid_h1 = T.dot(v0_sample, self.W) + self.hbias
        h1_mean = T.sigmoid(pre_sigmoid_h1)

        # get a sample of the hiddens given their activation
        h1_sample = self._rng.binomial(h1_mean.shape, p=h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # propagates the hidden units activation downwards to the visible units
        # Note that we return also the pre_sigmoid_activation of the
        # layer. As it will turn out later, due to how Theano deals with
        # optimizations, this symbolic variable will be needed to write
        # down a more stable computational graph (see details in the
        # reconstruction cost function)
        pre_sigmoid_v1 = T.dot(h0_sample, self.W.T) + self.vbias
        v1_mean = T.sigmoid(pre_sigmoid_v1)
        # get a sample of the visible given their activation
        v1_sample = self._rng.binomial(v1_mean.shape, p=v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def _get_pseudo_likelihood_cost(self, X, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = T.variable(value=0, dtype='int32', name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(X)

        # calculate free energy for the given bit configuration
        fe_xi = self._free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self._free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(
            self.n_visible * T.log(T.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def _get_reconstruction_cost(self, X, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """
        cross_entropy = T.mean(
            T.sum(
                X * T.log(T.sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - T.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy
