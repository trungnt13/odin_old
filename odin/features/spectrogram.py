# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram preprocess utilities is adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import six
import math

import numpy as np
import scipy.fftpack as fft
import scipy.signal

from . import preprocess as pp

MAX_MEM_BLOCK = 2**8 * 2**10
SMALL_FLOAT = 1e-20


def _validate_stft_arguments(n_fft, hop_length, win_length):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)
    return n_fft, hop_length, win_length


def fft_covert(fs, window=0.025, shift = 0.01):
    ''' Convert conventional information if window_length (in ms) and shift
    (in ms) to parameters for `stft` function

    Return
    ------
    n_fft: int
        number of fft should use
    hop_length: int
        number of sample points should be shifted
    win_length: int
        number of sample points for each windows
    '''
    window_length = int(round(window * fs))
    overlap = window_length - int(shift * fs)
    nfft = 2**int(np.ceil(np.log2(window_length)))
    return nfft, overlap, window_length


def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix, shape: [number_of_fft, time]


    Raises
    ------
    ValueError
        If `window` is supplied as a vector of length `n_fft`.


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> D
    array([[  2.576e-03 -0.000e+00j,   4.327e-02 -0.000e+00j, ...,
              3.189e-04 -0.000e+00j,  -5.961e-06 -0.000e+00j],
           [  2.441e-03 +2.884e-19j,   5.145e-02 -5.076e-03j, ...,
             -3.885e-04 -7.253e-05j,   7.334e-05 +3.868e-04j],
          ...,
           [ -7.120e-06 -1.029e-19j,  -1.951e-09 -3.568e-06j, ...,
             -4.912e-07 -1.487e-07j,   4.438e-06 -1.448e-05j],
           [  7.136e-06 -0.000e+00j,   3.561e-06 -0.000e+00j, ...,
             -5.144e-07 -0.000e+00j,  -1.514e-05 -0.000e+00j]], dtype=complex64)


    Use left-aligned frames, instead of centered frames


    >>> D_left = librosa.stft(y, center=False)


    Use a shorter hop length


    >>> D_short = librosa.stft(y, hop_length=64)


    Display a spectrogram


    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

    """

    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = pp.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        pp.valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = pp.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                     stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_.

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    Raises
    ------
    ValueError
        If `window` is supplied as a vector of length `n_fft`

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        ifft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = pp.pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_sum = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
        ifft_window_sum[sample:(sample + n_fft)] += ifft_window_square

    # Normalize by sum of squared window
    approx_nonzero_indices = ifft_window_sum > SMALL_FLOAT
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y


def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           norm=False, center=True, ref_power=1e-6, clip=True, dtype=np.complex64):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_.

    Calculates regular STFT as a side effect.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.

    norm : bool
        Normalize the STFT.

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
            `D[:, t]` (and `if_gram`) is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` at `y[t * hop_length]`

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating instantaneous frequency.
        Any bin with `np.abs(D[f, t])**2 < ref_power` will receive the
        default frequency estimate.

        If callable, the threshold is set to `ref_power(np.abs(D)**2)`.

    clip : boolean
        - If `True`, clip estimated frequencies to the range `[0, 0.5 * sr]`.
        - If `False`, estimated frequencies can be negative or exceed
          `0.5 * sr`.

    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    Returns
    -------
    if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
        Instantaneous frequency spectrogram:
        `if_gram[f, t]` is the frequency at bin `f`, time `t`

    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frequencies, D = librosa.ifgram(y, sr=sr)
    >>> frequencies
    array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
           [  3.150e+01,   3.070e+01, ...,   1.077e+01,   1.077e+01],
           ...,
           [  1.101e+04,   1.101e+04, ...,   1.101e+04,   1.101e+04],
           [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])

    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # Construct a padded hann window
    window = pp.pad_center(scipy.signal.hann(win_length, sym=False), n_fft)

    # Window for discrete differentiation
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)

    d_window = np.sin(-freq_angular) * np.pi / n_fft

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       window=window, center=center, dtype=dtype)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center, dtype=dtype).conj()

    # Compute power normalization. Suppress zeros.
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ValueError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1))
    bin_offset = (phase * diff_stft).imag / mag

    bin_offset[mag < ref_power**0.5] = 0

    if_gram = freq_angular[:n_fft // 2 + 1] + bin_offset

    if norm:
        stft_matrix = stft_matrix * 2.0 / window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix


def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram


    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ...,
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ...,
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)


    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ...,
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)

    """

    mag = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Examples
    --------
    >>> # Play at double speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
    >>> y_fast  = librosa.istft(D_fast, hop_length=512)

    >>> # Or play at 1/3 speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
    >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        STFT matrix

    rate :  float > 0 [scalar]
        Speed-up factor: `rate > 1` is faster, `rate < 1` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.

        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    Returns
    -------
    D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
        time-stretched STFT
    """

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        input spectrogram

    ref_power : scalar or function
        If scalar, `log(abs(S))` is compared to `log(ref_power)`.

        If a function, `log(abs(S))` is compared to `log(ref_power(abs(S)))`.

        This is primarily useful for comparing to the maximum value of `S`.

    amin    : float > 0[scalar]
        minimum amplitude threshold for `abs(S)` and `ref_power`

    top_db  : float >= 0 [scalar]
        threshold log amplitude at top_db below the peak:
        ``max(log(S)) - top_db``

    Returns
    -------
    log_S   : np.ndarray [shape=(d, t)]
        ``log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))``

    See Also
    --------
    perceptual_weighting

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.logamplitude(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.logamplitude(S**2, ref_power=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.logamplitude(S**2, ref_power=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()

    """

    if amin <= 0:
        raise ValueError('amin must be strictly positive')

    magnitude = np.abs(S)

    if six.callable(ref_power):
        # User supplied a function to calculate reference power
        __ref = ref_power(magnitude)
    else:
        __ref = np.abs(ref_power)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, __ref))

    if top_db is not None:
        if top_db < 0:
            raise ValueError('top_db must be non-negative positive')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


# ===========================================================================
# Filter
# ===========================================================================
def _fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreqs`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft // 2),
                       endpoint=True)


def _mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        number of Mel bins

    fmin      : float >= 0 [scalar]
        minimum frequency (Hz)

    fmax      : float >= 0 [scalar]
        maximum frequency (Hz)

    htk       : bool
        use HTK formula instead of Slaney

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = pp.hz_to_mel(fmin, htk=htk)
    max_mel = pp.hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return pp.mel_to_hz(mels, htk=htk)


def mel_filter(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = _fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = _mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - fftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    Examples
    --------
    >>> n_fft = 2048
    >>> dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
    >>> dct_filters
    array([[ 0.031,  0.031, ...,  0.031,  0.031],
           [ 0.044,  0.044, ..., -0.044, -0.044],
           ...,
           [ 0.044,  0.044, ..., -0.044, -0.044],
           [ 0.044,  0.044, ...,  0.044,  0.044]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(dct_filters, x_axis='linear')
    >>> plt.ylabel('DCT function')
    >>> plt.title('DCT filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)

    return basis


# ===========================================================================
# Spectrogram
# ===========================================================================
def spectrogram(y, n_fft=2048, hop_length=None, win_length=None, power=1):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.


    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, n_fft=n_fft, hop_length=hop_length)|**power`

    n_fft : int > 0
        - If `S` is provided, then `n_fft` is inferred from `S`
        - Else, copied from input
    '''
    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    # Otherwise, compute a magnitude spectrogram from input
    S = np.power(
        np.abs(
            stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)),
        power)

    return S


def ispectrogram(y, n_fft, hop_length=None, win_length=None, power=1,
    excit=None):
    ''' Attempt to go back from specgram-like powerspec to audio waveform
    by scaling specgram of white noise
    Parameters
    ----------
    y:
        y = abs(specgram(x*32768,NFFT,SAMPRATE,WINDOW,NOVERLAP)).^2;
    sr: int
        sample rate
    wintime: float
        windows time in millisecond
    steptime: float
        step time in millisecond
    '''
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # excit is input excitation; white noise is used if not specified

    #  for sr = 8000
    # NFFT = 256;
    # NOVERLAP = 120;
    # SAMPRATE = 8000;
    # WINDOW = hamming(200);
    n_fft, hop_length, win_length = _validate_stft_arguments(
        n_fft, hop_length, win_length)

    nrow, ncol = y.shape

    r = None
    if excit is not None:
        r = excit

    # Values coming out of rasta treat samples as integers,
    # not range -1..1, hence scale up here to match (approx)
    xlen = win_length + hop_length * (ncol - 1)

    if r is None:
        r = np.random.randn(xlen)
    R = stft(r, n_fft, hop_length=hop_length, win_length=win_length)
    if R.shape[1] < y.shape[1]:
        y = y[:, :R.shape[1]]
    elif R.shape[1] > y.shape[1]:
        R = R[:, :y.shape[1]]
    R = R * np.power(y, 1 / power)
    return istft(R, hop_length=hop_length, win_length=win_length)


def melspectrogram(y=None, sr=22050,
                   n_fft=2048, hop_length=None, win_length=None,
                   n_mels=128, fmin=0.0, fmax=None):
    """Compute a Mel-scaled power spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        power spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel power spectrogram

    See Also
    --------
    librosa.filters.mel
        Mel filter bank construction

    librosa.core.stft
        Short-time Fourier Transform


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
           [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
           ...,
           [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
           [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])

    Using a pre-computed power spectrogram

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D)

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.logamplitude(S,
    ...                                               ref_power=np.max),
    ...                          y_axis='mel', fmax=8000,
    ...                          x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Mel spectrogram')
    >>> plt.tight_layout()


    """

    S, n_fft = spectrogram(y=y, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, power=2)

    # Build a Mel filter
    mel_basis = mel_filter(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    return np.dot(mel_basis, S)


# -- Mel spectrogram and MFCCs -- #
def mfcc(y=None, sr=22050, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients

    Parameters
    ----------
    y     : np.ndarray [shape=(n,)] or None
        audio time series

    sr    : number > 0 [scalar]
        sampling rate of `y`

    S     : np.ndarray [shape=(d, t)] or None
        log-power Mel spectrogram

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    kwargs : additional keyword arguments
        Arguments to `melspectrogram`, if operating
        on time series input

    Returns
    -------
    M     : np.ndarray [shape=(n_mfcc, t)]
        MFCC sequence

    See Also
    --------
    melspectrogram

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
           [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
           ...,
           [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
           [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.logamplitude(S))
    array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
           [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
           ...,
           [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
           [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(mfccs, x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('MFCC')
    >>> plt.tight_layout()


    """
    S = logamplitude(melspectrogram(y=y, sr=sr, **kwargs))
    return np.dot(dct(n_mfcc, S.shape[0]), S)
