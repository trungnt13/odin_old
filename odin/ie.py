from __future__ import print_function, division

import os
from six.moves.urllib.request import FancyURLopener

from . import logger

# ======================================================================
# Net
# ======================================================================

class ParanoidURLopener(FancyURLopener):

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Exception('URL fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))

def get_file(fname, origin):
    ''' Get file from internet or local network.

    Parameters
    ----------
    fname : str
        name of downloaded file
    origin : str
        html link, path to file want to download

    Returns
    -------
    return : str
        path to downloaded file

    Notes
    -----
    Download files are saved at one of these location (order of priority):
     - ~/.dnntoolkit/datasets/
     - /tmp/.dnntoolkit/datasets/
    '''
    if os.path.exists(origin) and not os.path.isdir(origin):
        return origin
    import pwd
    user_name = pwd.getpwuid(os.getuid())[0]

    datadir_base = os.path.expanduser(os.path.join('~', '.dnntoolkit'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', user_name, '.dnntoolkit')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        logger.info('Downloading data from ' + origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            logger.progress(count * block_size, total_size,
                title='Downloading %s' % fname, newline=False,
                idx='downloading')

        ParanoidURLopener().retrieve(origin, fpath, dl_progress)
        progbar = None

    return fpath
