# -*- coding: utf-8 -*-
"""
Python module to help specify the source parameters.

.. module:: source

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

import os
import numpy as np
from ..headers import STF


def Mw(M0):
    """
    Calculate Moment Magnitude Mw from Seismic Moment M0 in N m
    """
    return 2. / 3 * (np.log10(M0) - 9.1)


def M0(Mw):
    """
    Calculate Seismic Moment M0 in N m from Moment Magnitude Mw
    """
    return 10**(1.5 * Mw + 9.1)


def source_frequency(fmax, stf='Gaussian'):
    """
    Calculate the angular frequency :math:`\omega`, ``freq`` attribute
    on the source line in the SW4 inputfile.

    Parameters
    ----------
    fmax : float
        Maximum frequency in the source-time function frequency content,
        Hz.

    stf : str
        Source-time function name. It can have the following values:
        'GaussianInt', 'Gaussian' (**default**), 'RickerInt', 'Ricker',
        'Ramp', 'Triangle', 'Sawtooth', 'Smoothwave', 'VerySmoothBump',
        'Brune', 'BruneSmoothed', 'GaussianWindow', 'Liu', 'Dirac', and
        'C6SmoothBump'.

        See the `SW4 User Guide
        <https://geodynamics.org/cig/software/sw4/>`_ for further
        details.

    Returns:
    2-tuple
        ``f0``, ``freq``. ``freq`` is the value which goes on the source
        line in the SW4 inputfile.
    """

    f0 = fmax / STF[stf].fmax2f0  # fundamental freqency, Hz
    freq = f0 * STF[stf].freq2f0  # angular frequency, rad/s

    return f0, freq


def t0(freq, t0=0.0, stf='Gaussian'):
    """
    Calculate the ``t0`` attribute on the source line in the SW4
    inputfile.

    Parameters
    ----------
    freq : float
        The angular frequency value used for the ``freq`` attribute on
        the source line in the SW4 inputfile.

    t0 : float
        The calculated ``t0`` is added to the supllied `t0` in the
        function call.

    stf : str
        The source-time function name.
    """
    if 'Gaussian' in stf:
        t0 += 6. / freq
    elif 'Ricker' in stf:
        t0 += 6. * 2 * np.pi / freq
    return t0 * 1.1  # take that extra 10%


def gaussian_stf(time, t0, freq):
    """
    Gaussian source-time-function.
    """
    return freq / np.sqrt(2 * np.pi) * np.exp(-0.5 * (freq * (time - t0)**2))


def f_max(vmin, h, ppw=8):
    """
    Calculate the maximum resolved frequency that meets the requirement
    that the shortest wavelength (:math:`\lambda=V_{min}/f_{max}`) be
    sampled by a minimum points-per-wavelength (`ppw`).

    Parameters
    ----------
    vmin : float
        Minimum seismic velocity in the computational doamin in m/s.

    h : float
        Grid spacing of the computational doamin in meters.

    ppw :
        Minimum points-per-wavelenght required for the computation. Set
        to 8 by default.

    Returns
    -------
    float
        The suggested ``fmax`` in Hz.

    See also
    --------
    .grid_spacing
    .get_vmin
    .source_frequency
    """
    return float(vmin) / (h * ppw)
