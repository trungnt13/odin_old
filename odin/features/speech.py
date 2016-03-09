from __future__ import print_function, division, absolute_import

import numpy as np

from collections import OrderedDict

# ==================== Predefined datasets information ==================== #
nist15_cluster_lang = OrderedDict([
    ['ara', ['ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb']],
    ['zho', ['zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu']],
    ['eng', ['eng-gbr', 'eng-usg', 'eng-sas']],
    ['fre', ['fre-waf', 'fre-hat']],
    ['qsl', ['qsl-pol', 'qsl-rus']],
    ['spa', ['spa-car', 'spa-eur', 'spa-lac', 'por-brz']]
])
nist15_lang_list = np.asarray([
    # Egyptian, Iraqi, Levantine, Maghrebi, Modern Standard
    'ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb',
    # Cantonese, Mandarin, Min Dong, Wu
    'zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu',
    # British, American, South Asian (Indian)
    'eng-gbr', 'eng-usg', 'eng-sas',
    # West african, Haitian
    'fre-waf', 'fre-hat',
    # Polish, Russian
    'qsl-pol', 'qsl-rus',
    # Caribbean, European, Latin American, Brazilian
    'spa-car', 'spa-eur', 'spa-lac', 'por-brz'])


def nist15_label(label):
    '''
    Return
    ------
    lang_id : int
        idx in the list of 20 language, None if not found
    cluster_id : int
        idx in the list of 6 clusters, None if not found
    within_cluster_id : int
        idx in the list of each clusters, None if not found
    '''
    label = label.replace('spa-brz', 'por-brz')
    rval = [None, None, None]
    # lang_id
    if label not in nist15_lang_list:
        raise ValueError('Cannot found label:%s' % label)
    rval[0] = np.argmax(label == nist15_lang_list)
    # cluster_id
    for c, x in enumerate(nist15_cluster_lang.iteritems()):
        j = x[1]
        if label in j:
            rval[1] = c
            rval[2] = j.index(label)
    return rval

# ==================== Timit ==================== #
timit_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th',
    'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
timit_39 = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd',
    'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't',
    'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
timit_map = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
    'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil'}


def timit_phonemes(phn, map39=False, blank=False):
    ''' Included blank '''
    if type(phn) not in (list, tuple, np.ndarray):
        phn = [phn]
    if map39:
        timit = timit_39
        timit_map = timit_map
        l = 39
    else:
        timit = timit_61
        timit_map = {}
        l = 61
    # ====== return phonemes ====== #
    rphn = []
    for p in phn:
        if p not in timit_map and p not in timit:
            if blank: rphn.append(l)
        else:
            rphn.append(timit.index(timit_map[p]) if p in timit_map else timit.index(p))
    return rphn


# ==================== Speech Signal Processing ==================== #
def read(f, pcm = False):
    '''
    Return
    ------
        waveform (ndarray), sample rate (int)
    '''
    if pcm or (isinstance(f, str) and 'pcm' in f):
        return np.memmap(f, dtype=np.int16, mode='r')

    from soundfile import read
    return read(f)


def preprocess(signal, add_noise=False):
    if len(signal.shape) > 1:
        signal = signal.ravel()
    signal = signal[signal != 0]
    signal = signal.astype(np.float32)
    if add_noise:
        signal = signal + 1e-13 * np.random.randn(signal.shape)
    return signal


def logmel(signal, fs, n_filters=40, n_ceps=13,
        win=0.025, shift=0.01,
        delta1=True, delta2=True, energy=False,
        normalize=True, clean=True):
    import sidekit

    if len(signal.shape) > 1:
        signal = signal.ravel()
    # 1. Some const.
    # n_filters = 40 # The number of mel filter bands
    f_min = 0. # The minimal frequency of the filter bank
    f_max = fs / 2
    # overlap = nwin - int(shift * fs)
    # 2. preprocess.
    if clean:
        signal = preprocess(signal)
    # 3. logmel.
    logmel = sidekit.frontend.features.mfcc(signal,
                    lowfreq=f_min, maxfreq=f_max,
                    nlinfilt=0, nlogfilt=n_filters,
                    fs=fs, nceps=n_ceps, midfreq=1000,
                    nwin=win, shift=shift,
                    get_spec=False, get_mspec=True)
    logenergy = logmel[1]
    logmel = logmel[3].astype(np.float32)
    # 4. delta.
    tmp = [logmel]
    if delta1 or delta2:
        d1 = sidekit.frontend.features.compute_delta(logmel,
                        win=3, method='filter')
        d2 = sidekit.frontend.features.compute_delta(d1,
                        win=3, method='filter')
        if delta1: tmp.append(d1)
        if delta2: tmp.append(d2)
    logmel = np.concatenate(tmp, 1)
    if energy:
        logmel = np.concatenate((logmel, logenergy.reshape(-1, 1)), axis=1)
    # 5. VAD and normalize.
    nwin = int(fs * win)
    idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
    # if not returnVAD:
        # logmel = logmel[idx, :]
    # Normalize
    if normalize:
        mean = np.mean(logmel, axis = 0)
        var = np.var(logmel, axis = 0)
        logmel = (logmel - mean) / np.sqrt(var)
    # return
    return logmel, idx


def mfcc(signal, fs, n_ceps, n_filters=40,
        win=0.025, shift=0.01,
        delta1=True, delta2=True, energy=False,
        normalize=True, clean=True):
    import sidekit

    # 1. Const.
    f_min = 0. # The minimal frequency of the filter bank
    f_max = fs / 2
    # 2. Speech.
    if clean:
        signal = preprocess(signal)
    #####################################
    # 3. mfcc.
    # MFCC
    mfcc = sidekit.frontend.features.mfcc(signal,
                    lowfreq=f_min, maxfreq=f_max,
                    nlinfilt=0, nlogfilt=n_filters,
                    fs=fs, nceps=n_ceps, midfreq=1000,
                    nwin=win, shift=shift,
                    get_spec=False, get_mspec=False)
    logenergy = mfcc[1]
    mfcc = mfcc[0].astype(np.float32)
    # 4. Add more information.
    tmp = [mfcc]
    if delta1 or delta2:
        d1 = sidekit.frontend.features.compute_delta(mfcc,
                        win=3, method='filter')
        d2 = sidekit.frontend.features.compute_delta(d1,
                        win=3, method='filter')
        if delta1: tmp.append(d1)
        if delta2: tmp.append(d2)
    mfcc = np.concatenate(tmp, 1)
    if energy:
        mfcc = np.concatenate((mfcc, logenergy.reshape(-1, 1)), axis=1)
    # 5. Vad and normalize.
    # VAD
    nwin = int(fs * win)
    idx = sidekit.frontend.vad.vad_snr(signal, 30, fs=fs, shift=shift, nwin=nwin)
    # if not returnVAD:
        # mfcc = mfcc[idx, :]
    # Normalize
    if normalize:
        mean = np.mean(mfcc, axis = 0)
        var = np.var(mfcc, axis = 0)
        mfcc = (mfcc - mean) / np.sqrt(var)
    # return
    return mfcc, idx


def spectrogram(signal, fs, n_ceps=13, n_filters=40,
        win=0.025, shift=0.01,
        delta1=True, delta2=True, energy=False,
        normalize=False, clean=True):
    import sidekit

    # 1. Const.
    f_min = 0. # The minimal frequency of the filter bank
    f_max = fs / 2
    # 2. Speech.
    if clean:
        signal = preprocess(signal)
    # 3. mfcc.
    # MFCC
    spt = sidekit.frontend.features.mfcc(signal,
                    lowfreq=f_min, maxfreq=f_max,
                    nlinfilt=0, nlogfilt=n_filters,
                    fs=fs, nceps=n_ceps, midfreq=1000,
                    nwin=win, shift=shift,
                    get_spec=True, get_mspec=False)
    logenergy = spt[1]
    spt = spt[2].astype(np.float32)
    # 4. Add more information.
    tmp = [spt]
    if delta1 or delta2:
        d1 = sidekit.frontend.features.compute_delta(spt,
                        win=3, method='filter')
        d2 = sidekit.frontend.features.compute_delta(d1,
                        win=3, method='filter')
        if delta1: tmp.append(d1)
        if delta2: tmp.append(d2)
    spt = np.concatenate(tmp, 1)
    if energy:
        spt = np.concatenate((spt, logenergy.reshape(-1, 1)), axis=1)

    # 5. Vad and normalize.
    # VAD
    nwin = int(fs * win)
    idx = sidekit.frontend.vad.vad_snr(signal, 30,
        fs=fs, shift=shift, nwin=nwin)
    # if not returnVAD:
        # spt = spt[idx, :]
    # Normalize
    if normalize:
        mean = np.mean(spt, axis = 0)
        var = np.var(spt, axis = 0)
        spt = (spt - mean) / np.sqrt(var)
    # return
    return spt, idx
