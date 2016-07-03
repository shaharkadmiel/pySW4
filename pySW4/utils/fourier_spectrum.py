"""
- fourier_spectrum.py -

Python wrapper for `numpy.fft` for conviniently computing the Fourier transform
of a signal.

By: Shahar Shani-Kadmiel, August 2015, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import sys
import numpy as np
import obspy

def fourier_spectrum(data, womean=False, winsize=None, stepsize=None,
                     delta=None, verbose=False):
    """Compute the Fourier spectrum of a signal either as a whole or as
    a series of windows with or without overlaps for smoother spectrum.

    The reason for windowing a signal into smaller chunks is to smooth a
    'noisy' Fourier spectrum of a very long signal. The frequency
    resolution of the FFT is related to the number of points (npts) passed
    into the FFT. The number of evenly spaced frequencies, sometimes refered
    to as bins, resolved between 0 and the nyquist frequency, 0.5/delta are
    (0.5*npts)+1 if npts is even or 0.5*(npts+1) if npts is odd.
    So the FFT of a signal of length 2048 npts and delta of 0.01 will have
    a nyquist frequency of 50 Hz and 1025 (aka FFT size or bins) resolved
    frequencies making the lowest resolved frequency 0.0488 Hz.
    Had the signal been longer, say 10000 points (100 seconds), the FFT size
    would be 5001 and the frequency resolution would be 0.01 Hz making the
    the spectrum very noisy numerically with many frequencies that don't
    contain any real information.

    Pay attention to the `stepsize` parameter. Leave it None if you are not
    sure what is the correct value. If a high energy peak in the time domain
    is picked up by more than one window due to ovelap, that can be accumulated
    to affect the over all amplitude in the frequency domain. On the other hand,
    if by slicing the signal into windows you introduce step-functions as the
    windowed signal starts with a steep rise or ends with a steep fall, the
    resulting frequency domain will contain false amplitudes at high frequencies
    as a result. Setting an overlap will get rid of this problem as these unwanted
    effects will be canceled out by one another or averaged by several windows.


    Params:
    -------

    data : either a sequence (say, values of a time-history) or an obspy
        Trace object. If a sequence is passed `delta` must be supplied.

    womean : remove mean before transform. Default is False, transform
        as is. This is usefull in cases that the signal does not revolve
        around zero but rather around some other constant value.

    winsize : by default `winsize` is None, taking the FFT of the entire
        signal as-is. Otherwise, `winsize` sets the size of the sliding
        window, taking `winsize` points of the signal at a time.
        Works fastest when `winsize` is a whole power of 2, i.e., 128, 512
        1024, 2048, 4096 etc.

    stepsize : the number of points by which the window slides each time.
        By default, `stepsize` is None, making it equal to `winsize`,
        no overlap. Setting `stepsize` to half `winsize` is common practice
        and will cause a 50% overlap.

    delta : the dt from one sample to the next such that 1/delta is the
        sampling rate.

    verbose : if True some information about the process is printed.

    Returns :
    ---------

    a frequency array and an amplitude array.
    """

    def _fft(signal, delta):
        freq = np.fft.rfftfreq(signal.size, delta)
        amp = np.abs(np.fft.rfft(signal))*delta
        return freq, amp

    # if data is a sequence
    if type(data) in [tuple,list,np.ndarray]:
        if verbose:
            message = """Processing data as a sequence..."""
            print(message)
            sys.stdout.flush()

        signal  = np.array(data)
        if delta is None:
            print('Error: If data is not an `obspy.core.trace.Trace` object\n'+\
                  '`delta` must be supplied.')
            return

    # if data is an obspy Trace object
    elif type(data) is obspy.core.trace.Trace:
        if verbose:
            message = """
Processing data as an obspy.core.trace.Trace object..."""
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
            message = """
Performing FFT on entire signal with %d point, no windoing..."""
            print(message %signal.size)
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
            print('Error: stepsize must be smaller than or equal to winsize')
            return
        if winsize > signal.size:
            print('Error: winsize must be smaller than or equal to npts')
            return

        overlap = winsize - stepsize
        n_of_win = signal.size//(winsize - overlap)

        if verbose:
            message = """
Performing FFT on %d long signal with %d windows of size %d points
and %d points overlap.
May take a while if winsize or stepsize are small..."""
            print(message %(signal.size, n_of_win, winsize, overlap))
            sys.stdout.flush()

        # pad the end of the signal with zeros
        # to make sure the entire signal is used
        # for the fft.
        padded = np.pad(signal,(0,winsize),mode='constant')
        freq = np.fft.rfftfreq(winsize, delta)
        amps = np.zeros_like(freq)

        for i in xrange(n_of_win):
            if verbose:
                print('\rFFT window %d out of %d...' %(i+1, n_of_win)),
                sys.stdout.flush()

            start = i*stepsize
            stop = start + winsize
            amps += np.abs(np.fft.rfft(padded[start:stop]))*delta

        amp = amps/n_of_win
        return freq, amp
