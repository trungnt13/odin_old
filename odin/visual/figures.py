from __future__ import print_function, absolute_import, division

import numpy as np
from six.moves import zip, range
from .. import logger

# ===========================================================================
# Helpers
# From DeepLearningTutorials: http://deeplearning.net
# ===========================================================================
def resize_images(x, shape):
    from scipy.misc import imresize

    reszie_func = lambda x, shape: imresize(x, shape, interp='bilinear')
    if x.ndim == 4:
        def reszie_func(x, shape):
            # x: 3D
            # The color channel is the first dimension
            tmp = []
            for i in x:
                tmp.append(imresize(i, shape).reshape((-1,) + shape))
            return np.swapaxes(np.vstack(tmp).T, 0, 1)

    imgs = []
    for i in x:
        imgs.append(reszie_func(i, shape))
    return imgs

def tile_raster_images(X, tile_shape=None, tile_spacing=(2, 2), spacing_value=0.):
    ''' This function create tile of images

    Parameters
    ----------
    X : 3D-gray or 4D-color images
        for color images, the color channel must be the second dimension
    tile_shape : tuple
        resized shape of images
    tile_spacing : tuple
        space betwen rows and columns of images
    spacing_value : int, float
        value used for spacing

    '''
    if X.ndim == 3:
        img_shape = X.shape[1:]
    elif X.ndim == 4:
        img_shape = X.shape[2:]
    else:
        raise ValueError('Unsupport %d dimension images' % X.ndim)
    if tile_shape is None:
        tile_shape = img_shape
    if tile_spacing is None:
        tile_spacing = (2, 2)

    if img_shape != tile_shape:
        X = resize_images(X, tile_shape)
    else:
        X = [np.swapaxes(x.T, 0, 1) for x in X]

    n = len(X)
    n = int(np.ceil(np.sqrt(n)))

    # create spacing
    rows_spacing = np.zeros_like(X[0])[:tile_spacing[0], :] + spacing_value
    nothing = np.vstack((np.zeros_like(X[0]), rows_spacing))
    cols_spacing = np.zeros_like(nothing)[:, :tile_spacing[1]] + spacing_value

    # ====== Append columns ====== #
    rows = []
    for i in range(n): # each rows
        r = []
        for j in range(n): # all columns
            idx = i * n + j
            if idx < len(X):
                r.append(np.vstack((X[i * 4 + j], rows_spacing)))
            else:
                r.append(nothing)
            if j != n - 1:   # cols spacing
                r.append(cols_spacing)
        rows.append(np.hstack(r))
    # ====== Append rows ====== #
    img = np.vstack(rows)[:-tile_spacing[0]]
    return img

# ===========================================================================
# Plotting methods
# ===========================================================================
def plot_images(x, tile_shape=None, tile_spacing=None,
    fig=None, path=None, show=False):
    '''
    x : 2D-gray or 3D-color images
        for color image the color channel is second dimension
    '''
    from matplotlib import pyplot as plt
    if x.ndim == 3 or x.ndim == 2:
        cmap = plt.cm.Greys_r
    elif x.ndim == 4:
        cmap = None
    else:
        raise ValueError('NO support for %d dimensions image!' % x.ndim)

    x = tile_raster_images(x, tile_shape, tile_spacing)
    if fig is None:
        fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    subplot.imshow(x, cmap=cmap)
    subplot.axis('off')

    if path:
        plt.savefig(path, dpi=300, format='png', bbox_inches='tight')
    if show:
        plt.show(block=False)
        raw_input('<Enter> to close the figure ...')
    else:
        return fig

def plot_images_old(x, fig=None, titles=None, path=None, show=False):
    '''
    x : 2D-gray or 3D-color images
        for color image the color channel is second dimension
    '''
    from matplotlib import pyplot as plt
    if x.ndim == 3 or x.ndim == 2:
        cmap = plt.cm.Greys_r
    elif x.ndim == 4:
        cmap = None
        shape = x.shape[2:] + (x.shape[1],)
        x = np.vstack([i.T.reshape((-1,) + shape) for i in x])
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

    if show:
        plt.show(block=False)
        raw_input('<Enter> to close the figure ...')
    else:
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

# ===========================================================================
# Helper methods
# ===========================================================================
def plot_show():
    from matplotlib import pyplot as plt
    plt.show(block=False)
    raw_input('<enter> to close all plots')
    plt.close('all')

def plot_save(path, figs=None, dpi=300):
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        pp = PdfPages(path)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        logger.info('Saved pdf figures to:%s' % str(path))
    except Exception, e:
        logger.error('Cannot save figures to pdf, error:%s' % str(e))
