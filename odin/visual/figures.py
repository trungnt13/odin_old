from __future__ import print_function, absolute_import, division

from matplotlib import pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, labels, axis=None, fontsize=13):
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

def plot_weights(x, ax=None, colormap = "Greys", colorbar=False, path=None, keep_aspect=True):
    '''
    Parameters
    ----------
    x : numpy.ndarray
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
    for i in xrange(nrows):
        for j in xrange(ncols):
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
