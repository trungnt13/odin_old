from __future__ import print_function, division, absolute_import

import os
from stat import S_ISDIR
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
     - ~/.odin/datasets/
     - /tmp/.odin/datasets/
    '''
    if os.path.exists(origin) and not os.path.isdir(origin):
        return origin
    import pwd
    user_name = pwd.getpwuid(os.getuid())[0]

    datadir_base = os.path.expanduser(os.path.join('~', '.odin'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', user_name, '.odin')
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


class SSH(object):

    """Create a SSH connection object
        Example:
            ssh = SSH('192.168.1.16',username='user',password='pass')
            ssh.ls('.') # same as ls in linux
            ssh.open('/path/to/file') # open stream to any file in remote server
            ssh.get_file('/path/to/file') # read the whole file in remote server
            ssh.close()
    """

    def __init__(self, hostname, username, password=None, pkey_path=None, port=22):
        super(SSH, self).__init__()
        import paramiko

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        k = None
        if pkey_path:
            k = paramiko.RSAKey.from_private_key_file(pkey_path)
        ssh.connect(hostname=hostname,
                    username=username, port=port,
                    password=password,
                    pkey = k)
        sftp = ssh.open_sftp()
        self.ssh = ssh
        self.sftp = sftp

    def _file_filter(self, fname):
        if len(fname) == 0 or fname == '.' or fname == '..':
            return False
        return True

    def ls(self, path='.'):
        sin, sout, serr = self.ssh.exec_command('ls -a ' + path)
        file_list = sout.read()
        file_list = [f for f in file_list.split('\n') if self._file_filter(f)]
        return file_list

    def open(self, fpaths, mode='r', bufsize=-1):
        if not (isinstance(fpaths, list) or isinstance(fpaths, tuple)):
            fpaths = [fpaths]
        results = []
        for f in fpaths:
            try:
                results.append(self.sftp.open(f, mode=mode, bufsize=bufsize))
            except:
                pass
        if len(results) == 1:
            return results[0]
        return results

    def get_file(self, fpaths, bufsize=-1):
        if not (isinstance(fpaths, list) or isinstance(fpaths, tuple)):
            fpaths = [fpaths]
        results = []
        for f in fpaths:
            try:
                results.append(self.sftp.open(f, mode='r', bufsize=bufsize))
            except:
                pass
        if len(results) == 1:
            return results[0]
        return results

    def isdir(self, path):
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            #Path does not exist, so by definition not a directory
            return None

    def getwd(self):
        ''' This method may return NONE '''
        return self.sftp.getcwd()

    def setwd(self, path):
        self.sftp.chdir(path)

    def mkdir(self, path, mode=511):
        self.sftp.mkdir(path, mode)

    def close(self):
        self.sftp.close()
        self.ssh.close()
