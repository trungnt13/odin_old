from __future__ import print_function, absolute_import, division

import numpy as np
from six.moves import zip, range

# ===========================================================================
# Helpers
# From DeepLearningTutorials: http://deeplearning.net
# ===========================================================================
def _scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    Parameters
    ----------
    X : a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None
        a 2-D array in which every row is a flattened image.

    img_shape : tuple; (height, width)
        the original shape of each image

    tile_shape: tuple; (rows, cols)
        the number of images to tile (rows, cols)

    output_pixel_vals : bool
        if output should be pixel values (i.e. int8 values) or floats

    scale_rows_to_unit_interval: bool
        if the values need to be scaled before being plotted to [0,1] or not


    Returns
    -------
    a 2-d array with same dtype as X, array suitable for viewing as an image.
    (See:`Image.fromarray`.)


    Example
    -------

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = _scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array

# ===========================================================================
# Plotting
# ===========================================================================
def plot_images(x, fig=None, titles=None, path=None):
    from matplotlib import pyplot as plt
    if x.ndim == 3 or x.ndim == 2:
        cmap = plt.cm.Greys_r
    elif x.ndim == 4:
        cmap = None
    else:
        raise ValueError('NO support for %d dimensions image!' % x.ndim)

    if x.ndim == 2:
        ncols = 1
        nrows = 1
    else:
        ncols = int(np.ceil(np.sqrt(x.shape[0])))
        nrows = int(ncols)

    if fig is None:
        fig = plt.figure()
    if titles is not None:
        if not isinstance(titles, (tuple, list)):
            titles = [titles]
        if len(titles) != x.shape(0):
            raise ValueError('Titles must have the same length with \
                the number of images!')

    for i in range(ncols):
        for j in range(nrows):
            idx = i * ncols + j
            if idx < x.shape[0]:
                subplot = fig.add_subplot(nrows, ncols, idx + 1)
                subplot.imshow(x[idx], cmap=cmap)
                if titles is not None:
                    subplot.set_title(titles[idx])
                subplot.axis('off')
    if path:
        plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    return fig

def plot_confusion_matrix(cm, labels, axis=None, fontsize=13):
    from matplotlib import pyplot as plt

    title = 'Confusion matrix'
    cmap = plt.cm.Blues

    # column normalize
    if np.max(cm) > 1:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm
    if axis is None:
        axis = plt.gca()

    im = axis.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    axis.set_title(title)
    axis.get_figure().colorbar(im)

    tick_marks = np.arange(len(labels))
    axis.set_xticks(tick_marks)
    axis.set_yticks(tick_marks)
    axis.set_xticklabels(labels, rotation=90, fontsize=13)
    axis.set_yticklabels(labels, fontsize=13)
    axis.set_ylabel('True label')
    axis.set_xlabel('Predicted label')
    # axis.tight_layout()
    return axis

def plot_weights(x, ax=None, colormap = "Greys", colorbar=False, path=None,
    keep_aspect=True):
    '''
    Parameters
    ----------
    x : np.ndarray
        2D array
    ax : matplotlib.Axis
        create by fig.add_subplot, or plt.subplots
    colormap : str
        colormap alias from plt.cm.Greys = 'Greys'
    colorbar : bool, 'all'
        whether adding colorbar to plot, if colorbar='all', call this
        methods after you add all subplots will create big colorbar
        for all your plots
    path : str
        if path is specified, save png image to given path

    Notes
    -----
    Make sure nrow and ncol in add_subplot is int or this error will show up
     - ValueError: The truth value of an array with more than one element is
        ambiguous. Use a.any() or a.all()

    Example
    -------
    >>> x = np.random.rand(2000, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(2, 2, 1)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 2)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 3)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 4)
    >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
    >>> plt.show()
    '''
    from matplotlib import pyplot as plt

    if colormap is None:
        colormap = plt.cm.Greys

    if x.ndim > 2:
        raise ValueError('No support for > 2D')
    elif x.ndim == 1:
        x = x[:, None]

    ax = ax if ax is not None else plt.gca()
    if keep_aspect:
        ax.set_aspect('equal', 'box')
    # ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(str(x.shape), fontsize=6)
    img = ax.pcolorfast(x, cmap=colormap, alpha=0.8)
    plt.grid(True)

    if colorbar == 'all':
        fig = ax.get_figure()
        axes = fig.get_axes()
        fig.colorbar(img, ax=axes)
    elif colorbar:
        plt.colorbar(img, ax=ax)

    if path:
        plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    return ax

def plot_weights3D(x, colormap = "Greys", path=None):
    '''
    Example
    -------
    >>> # 3D shape
    >>> x = np.random.rand(32, 28, 28)
    >>> dnntoolkit.visual.plot_conv_weights(x)
    >>> # 4D shape
    >>> x = np.random.rand(32, 3, 28, 28)
    >>> dnntoolkit.visual.plot_conv_weights(x)
    '''
    from matplotlib import pyplot as plt

    if colormap is None:
        colormap = plt.cm.Greys

    shape = x.shape
    if len(shape) == 3:
        ncols = int(np.ceil(np.sqrt(shape[0])))
        nrows = int(ncols)
    elif len(shape) == 4:
        ncols = shape[0]
        nrows = shape[1]
    else:
        raise ValueError('Unsupport for %d dimension' % x.ndim)

    fig = plt.figure()
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            count += 1
            # skip
            if x.ndim == 3 and count > shape[0]:
                continue

            ax = fig.add_subplot(nrows, ncols, count)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0 and j == 0:
                ax.set_xlabel('New channels', fontsize=6)
                ax.xaxis.set_label_position('top')
                ax.set_ylabel('Old channels', fontsize=6)
                ax.yaxis.set_label_position('left')
            else:
                ax.axis('off')
            # image data
            if x.ndim == 4:
                img = ax.pcolorfast(x[j, i], cmap=colormap, alpha=0.8)
            else:
                img = ax.pcolorfast(x[count - 1], cmap=colormap, alpha=0.8)
            plt.grid(True)

    # colorbar
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)

    if path:
        plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    return fig

def plot_hinton(matrix, max_weight=None, ax=None):
    '''
    Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
    a weight matrix):
        Positive: white
        Negative: black
    squares, and the size of each square represents the magnitude of each value.
    * Note: performance significant decrease as array size > 50*50
    Example:
        W = np.random.rand(10,10)
        hinton_plot(W)
    '''
    from matplotlib import pyplot as plt

    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    return ax
