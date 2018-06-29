"""
Python module for spectral analysis.

.. module:: spectral

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import sys
import numpy as np
from matplotlib.mlab import psd
import obspy


def next_power_2(x):
    return 1 << (x - 1).bit_length()


def psde(data, womean=False, winsize='default', stepsize=None,
         delta=None, verbose=False):
    """
    Wrapper for :func:`matplotlib.mlab.psd` for conveniently estimating
    the power spectral density of a signal.

    See :func:`~pySW4.utils.spectral.fourier_spectrum` documentation for
    keyword argument explanation and rational.

    Returns
    -------
    2 :class:`~numpy.ndarray`

        - frequency array and
        - amplitude array
    """

    # if data is a sequence
    if type(data) in [tuple, list, np.ndarray]:
        if verbose:
            message = 'Processing data as a sequence...'
            print(message)
            sys.stdout.flush()

        signal = np.array(data)
        if delta is None:
            msg = ('If data is not an `obspy.core.trace.Trace` object\n'
                   '``delta`` must be supplied.')
            raise ValueError(msg)

    # if data is an obspy Trace object
    elif type(data) is obspy.core.trace.Trace:
        if verbose:
            message = ('Processing data as an `obspy.core.trace.Trace` '
                       'object...')
            print(message)
            sys.stdout.flush()

        signal = data.data
        delta = data.stats.delta

    # pad the signal and fft all of it, no windoing...
    if winsize is 'default':
        winsize = next_power_2(signal.size)
        noverlap = 0
        if verbose:
            message = ('Performing PSDE on padded signal with %d points, '
                       'no windoing...')
            print(message % winsize)
            sys.stdout.flush()

    # fft the entire signal, no windoing...
    elif winsize is None:
        winsize = signal.size
        noverlap = 0
        if verbose:
            message = ('Performing PSDE on entire signal with %d points, no '
                       'windoing...')
            print(message % winsize)
            sys.stdout.flush()

    # cut the signal into chuncks
    else:
        winsize = int(winsize)
        if stepsize is None:
            stepsize = winsize

        noverlap = winsize - stepsize
        if noverlap < 0:
            msg = '``stepsize`` must be smaller than or equal to ``winsize``.'
            raise ValueError(msg)

        n_of_win = signal.size // (winsize - noverlap)

        if verbose:
            message = ('Performing PSDE on %d long signal with %d windows of '
                       'size %d points and %d points overlap. May take a '
                       'while if ``winsize`` or ``stepsize`` are small...')
            print(message % (signal.size, n_of_win, winsize, noverlap))
            sys.stdout.flush()

    if womean:
        detrend = 'mean'
    else:
        detrend = 'none'

    amp, freq = psd(signal, NFFT=winsize, Fs=1 / delta, detrend=detrend,
                    noverlap=noverlap)
    return freq, amp


def fourier_spectrum(data, womean=False, winsize=None, stepsize=None,
                     delta=None, verbose=False):
    """
    Compute the Fourier spectrum of a signal either as a whole or as
    a series of windows with or without overlaps for smoother spectrum.

    The reason for windowing a signal into smaller chunks is to smooth a
    'noisy' Fourier spectrum of a very long signal. The frequency
    resolution of the FFT is related to the number of points (npts)
    passed into the FFT. The number of evenly spaced frequencies,
    sometimes refered to as bins, resolved between 0 and the nyquist
    frequency, 0.5/delta are (0.5*npts)+1 if npts is even or
    0.5*(npts+1) if npts is odd.

    So the FFT of a signal of length 2048 npts and delta of 0.01 will
    have a nyquist frequency of 50 Hz and 1025 (aka FFT size or bins)
    resolved frequencies making the lowest resolved frequency 0.0488 Hz.
    Had the signal been longer, say 10000 points (100 seconds), the FFT
    size would be 5001 and the frequency resolution would be 0.01 Hz
    making the the spectrum very noisy numerically with many frequencies
    that don't contain any real information.

    Pay attention to the ``stepsize`` parameter. Leave it ``None`` if
    you are not sure what is the correct value. If a high energy peak in
    the time domain is picked up by more than one window due to ovelap,
    that can be accumulated to affect the over all amplitude in the
    frequency domain. On the other hand, if by slicing the signal into
    windows you introduce step-functions as the windowed signal starts
    with a steep rise or ends with a steep fall, the resulting frequency
    spectrum will contain false amplitudes at high frequencies as a
    result. Setting an overlap will get rid of this problem as these
    unwanted effects will be canceled out by one another or averaged by
    several windows.

    .. note:: This function is a convinient wrapper for the
              :func:`~numpy.fft` function.

    Parameters
    ----------
    data : :class:`~numpy.ndarray` or :class:`~obspy.core.trace.Trace` instance
        If a sequence (:class:`~numpy.ndarray`) of time-history values
        is passed ``delta`` must be supplied as well. Otherwise pass an
        :class:`~obspy.core.trace.Trace` instance.

    womean : bool
        Remove the mean of the signal before performing the transform.
        Default is False, transform as is. In cases where there is a
        shift in the signal such that the baseline is not zero, setting
        ``womean=True`` will result in a smaller DC.

    winsize : int
        By default `winsize` is None, taking the FFT of the entire
        signal as-is. Otherwise, `winsize` sets the size of the
        sliding window, taking `winsize` points of the signal at a
        time. Works fastest when `winsize` is a whole power of 2,
        i.e., 128, 512, 1024, 2048, 4096 etc.

    stepsize : int
        The number of points by which the window slides each time.
        By default, `stepsize` is None, making it equal to
        `winsize`, no overlap. Setting `stepsize` to half
        `winsize` is common practice and will cause a 50% overlap.

    delta : float
        The dt from one sample to the next such that 1/delta is the
        sampling rate.

    verbose : bool
        If set to ``True`` some information about the process is
        printed.

    Returns
    -------
    2 :class:`~numpy.ndarray`

        - frequency array and
        - amplitude array
    """

    def _fft(signal, delta):
        freq = np.fft.rfftfreq(signal.size, delta)
        amp = np.abs(np.fft.rfft(signal)) * delta
        return freq, amp

    # if data is a sequence
    if type(data) in [tuple, list, np.ndarray]:
        if verbose:
            message = 'Processing data as a sequence...'
            print(message)
            sys.stdout.flush()

        signal = np.array(data)
        if delta is None:
            msg = ('If data is not an `obspy.core.trace.Trace` object\n'
                   '``delta`` must be supplied.')
            raise ValueError(msg)

    # if data is an obspy Trace object
    elif type(data) is obspy.core.trace.Trace:
        if verbose:
            message = ('Processing data as an `obspy.core.trace.Trace` '
                       'object...')
            print(message)
            sys.stdout.flush()

        signal = data.data
        # if womean is not False, remove the mean befor transform
        if womean:
            signal -= signal.mean()
        delta = data.stats.delta

    # fft the entire signal, no windoing...
    if winsize is None:
        if verbose:
            message = ('Performing FFT on entire signal with %d points, '
                       'no windoing...')
            print(message % signal.size)
            sys.stdout.flush()

        return _fft(signal, delta)

    # cut the signal into overlaping windows,
    # fft each window and average the sum
    else:
        winsize = int(winsize)
        if stepsize is None:
            stepsize = winsize
        else:
            stepsize = int(stepsize)

        if stepsize > winsize:
            msg = '``stepsize`` must be smaller than or equal to ``winsize``.'
            raise ValueError(msg)
        if winsize > signal.size:
            msg = '``winsize`` must be smaller than or equal to ``npts``.'
            raise ValueError(msg)

        overlap = winsize - stepsize
        n_of_win = signal.size // (winsize - overlap)

        if verbose:
            message = ('Performing FFT on %d long signal with %d windows of '
                       'size %d points and %d points overlap. May take a '
                       'while if ``winsize`` or ``stepsize`` are small...')
            print(message % (signal.size, n_of_win, winsize, overlap))
            sys.stdout.flush()

        # pad the end of the signal with zeros
        # to make sure the entire signal is used
        # for the fft.
        padded = np.pad(signal, (0, winsize), mode='constant')
        freq = np.fft.rfftfreq(winsize, delta)
        amps = np.zeros_like(freq)

        for i in xrange(n_of_win):
            if verbose:
                print('\rFFT window %d out of %d...' % (i + 1, n_of_win)),
                sys.stdout.flush()

            start = i * stepsize
            stop = start + winsize
            amps += np.abs(np.fft.rfft(padded[start:stop])) * delta

        amp = amps / n_of_win
        return freq, amp
