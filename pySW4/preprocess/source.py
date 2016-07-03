# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Helper module for specifying the source for a WPP/SW4 run
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------

"""
- source.py -

Python module to help specifying the source for a WPP/SW4 run

By: Shahar Shani-Kadmiel, September 2015, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import os
import numpy as np

def Mw(M0):
    """ Calculate Moment Magnitude Mw from Seismic Moment M0 in N m"""

    return 2./3. * (np.log10(self.M0) - 9.1)


def M0(Mw):
    """ Calculate Seismic Moment M0 in N m from Moment Magnitude Mw"""
    return 10**(1.5*Mw + 9.1)

def source_frequency(fmax, t0=0):
    """Calculate the angular frequency omega (``freq``) which goes on the source
    command of the WPP inputfile for Gaussian and GaussianInt Source
    Time Functions (GSTF/GISTF). Calculate the optimal t0 parameter for
    the STF. t0 must be equal to or greater then this optimal value and STF[0]
    should be < 1e-8.
    fmax is in Hz

    returns: f0, omega, t0"""

    f0 = fmax/2.5 # fundamental freqency, Hz
    omega = f0*2*np.pi  # angular frequency, rad/s

    t0 += 6./omega
    # stf0 = gstf(0, t0, omega)
    # while stf0 >= 10**-8:
    #   t0 += 1/omega
    #   stf0 = gstf(0, t0, omega)

    return f0, omega, t0


def gaussian_stf(time, t0, omega):
    """Gaussian source-time-function"""

    return omega/np.sqrt(2*np.pi)*np.exp(-0.5*(omega*(time-t0)**2))


def f_max(vmin, h, ppw=15):  # Note: ppw is regarded differently in WPP and SW4
    """Calculate teh maximum resolved frequency as a function of the
    minimum wave velocity, the grid spacing and the number of points
    per wavelength."""
    return float(vmin) / (h * ppw)


def f0(fmax, source_type):
    """Calculate the fundamental frequency f_0 based on fmax and the
    source type"""
    if source_type in ['Ricker', 'RickerInt', 'Gaussian', 'GaussianInt']:
        f_0 = fmax / 2.5
    elif source_type in ['Brune', 'BruneSmoothed']:
        f_0 = fmax / 4
    return f_0


def omega(f0, source_type):
    """Calculate omega, that value that goes on the source line in the
    WPP input file as ``freq`` based on f_0 and the source type"""
    if source_type in ['Ricker', 'RickerInt']:
        freq = f0
    elif source_type in ['Brune', 'BruneSmoothed', 'Gaussian', 'GaussianInt']:
        freq = f0 * 2 * np.pi
    return freq
