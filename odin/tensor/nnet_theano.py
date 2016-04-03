from __future__ import division

import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d

from . import theano_backend as TB
from .. import config

_FLOATX = config.floatX()
_EPSILON = config.epsilon()

if TB.on_gpu():
    from theano.sandbox.cuda import dnn


# ===========================================================================
# NN OPERATIONS
# ===========================================================================
def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    x = T.nnet.relu(x, alpha)
    if max_value is not None:
        x = T.minimum(x, max_value)
    return x


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def linear(x):
    return x


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def tanh(x):
    return T.tanh(x)


# ==================== Regularizations ==================== #
def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm


def l2_regularize(x):
    return T.sum(T.square(x))


def l1_regularize(x):
    return T.sum(T.abs_(x))


def jacobian_regularize(hidden, params):
    ''' Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    '''
    hidden = hidden * (1 - hidden)
    L = TB.expand_dims(hidden, 1) * TB.expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = T.sum(T.pow(L, 2)) / hidden.shape[0]
    return T.mean(L)


# ===========================================================================
# CONVOLUTIONS
# ===========================================================================
def conv2d(x, kernel, strides=(1, 1),
           border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])

    if TB.on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            np_kernel = kernel.eval()
            # mode same and even filter
            if len([s for s in np_kernel.shape[2:] if s % 2 == 0]) > 0.:
                assert strides[0] <= np_kernel.shape[2], \
                    'strides should be smaller than the convolution window.'
                assert strides[1] <= np_kernel.shape[3], \
                    'strides should be smaller than the convolution window.'
                conv_out = dnn.dnn_conv(img=x,
                                        kerns=kernel,
                                        border_mode='full')
                shift_x = (np_kernel.shape[2] - strides[0]) // 2
                shift_y = (np_kernel.shape[3] - strides[1]) // 2
                expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
                expected_height = (x.shape[3] + strides[1] - 1) // strides[1]
                conv_out = conv_out[:, :,
                                    shift_x: shift_x + expected_width,
                                    shift_y: shift_y + expected_height]
            else: # same mode and odd filter
                border_mode = tuple(s // 2 for s in np_kernel.shape[2:])
                # only use float32 for dnn.dnn_conv
                conv_out = dnn.dnn_conv(img=x.astype('float32'),
                                        kerns=kernel.astype('float32'),
                                        border_mode=border_mode,
                                        subsample=strides).astype(x.dtype)
        else:
            conv_out = dnn.dnn_conv(img=x.astype('float32'),
                                    kerns=kernel.astype('float32'),
                                    border_mode=border_mode,
                                    subsample=strides).astype(x.dtype)
    else:
        if border_mode == 'same' or border_mode == 'full':
            th_border_mode = 'full'
            np_kernel = kernel.eval()
            assert strides[0] <= np_kernel.shape[2], 'strides should be smaller than the convolution window.'
            assert strides[1] <= np_kernel.shape[3], 'strides should be smaller than the convolution window.'
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        elif isinstance(border_mode, (tuple, list)):
            th_border_mode = border_mode
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape)
        if border_mode == 'same':
            shift_x = (np_kernel.shape[2] - strides[0]) // 2
            shift_y = (np_kernel.shape[3] - strides[1]) // 2
            expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
            expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

            conv_out = conv_out[:, :,
                                shift_x: shift_x + expected_width,
                                shift_y: shift_y + expected_height]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def deconv2d(x, kernel, img_shape,
    strides=(1, 1), border_mode='valid', dim_ordering='th'):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    img_shape: (width, height) of original image
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.dimshuffle((0, 3, 1, 2))
        kernel = kernel.dimshuffle((3, 2, 0, 1))

    border_mode = 'half' if border_mode == 'same' else border_mode
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
        imshp=None,
        kshp=None,
        subsample=strides, border_mode=border_mode,
        filter_flip=True)
    # only support float64 ops
    conv_out = op(kernel.astype('float32'),
                  x.astype('float32'),
                  img_shape).astype(x.dtype)

    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    conv_mode: string, "conv" or "cross".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, time)
        # TH kernel shape: (depth, input_depth, rows, cols, time)
        # TF input shape: (samples, rows, cols, time, input_depth)
        # TF kernel shape: (rows, cols, time, input_depth, depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
        if image_shape:
            image_shape = (image_shape[0], image_shape[4],
                           image_shape[1], image_shape[2],
                           image_shape[3])
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1],
                            filter_shape[2])

    if TB.on_gpu() and dnn.dnn_available():
        if border_mode == 'same':
            np_kernel = kernel.eval()
            border_mode = tuple(s // 2 for s in np_kernel.shape[2:])
        # only use float32 for dnn.dnn_conv3d
        conv_out = dnn.dnn_conv3d(img=x.astype('float32'),
                                  kerns=kernel.astype('float32'),
                                  border_mode=border_mode,
                                  subsample=strides).astype(x.dtype)
    else:
        if border_mode == 'same':
            assert(strides == (1, 1, 1))
            pad_dim1 = (kernel.shape[2] - 1)
            pad_dim2 = (kernel.shape[3] - 1)
            pad_dim3 = (kernel.shape[4] - 1)
            output_shape = (x.shape[0], x.shape[1],
                            x.shape[2] + pad_dim1,
                            x.shape[3] + pad_dim2,
                            x.shape[4] + pad_dim3)
            output = T.zeros(output_shape)
            indices = (slice(None), slice(None),
                       slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                       slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                       slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
            x = T.set_subtensor(output[indices], x)
            border_mode = 'valid'

        border_mode_3d = (border_mode, border_mode, border_mode)
        conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                                   filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                                   border_mode=border_mode_3d)
        conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

        # support strides by manually slicing the output
        if strides != (1, 1, 1):
            conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 3, 1, 2))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    # ====== pooling ====== #
    if TB.on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else: # CPU veresion support by theano
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    # ====== dim ordering ====== #
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))
    # ====== border mode ====== #
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        d_pad = pool_size[2] - 2 if pool_size[2] % 2 == 1 else pool_size[2] - 1
        padding = (w_pad, h_pad, d_pad)
    elif border_mode == 'valid':
        padding = (0, 0, 0)
    elif isinstance(border_mode, (tuple, list)):
        padding = tuple(border_mode)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))
    # ====== pooling ====== #
    if TB.on_gpu() and dnn.dnn_available():
        pool_out = dnn.dnn_pool(x, pool_size,
                                stride=strides,
                                mode=pool_mode,
                                pad=padding)
    else:
        padding = padding[:2]
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=True,
                              padding=padding,
                              mode=pool_mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=True,
                                padding=padding,
                                mode=pool_mode)
    # ====== output ====== #
    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out
