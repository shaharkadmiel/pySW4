# -*- coding: utf-8 -*-
"""
Python module with general utilities for making plotting easier.

.. module:: utils

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

from matplotlib import rc
import numpy as np


def pretty_ticks(min, max, number_of_ticks=2, show_zero=False):
    """
    Function for controlling tickmarks.

    Parameters
    ----------
    min : int or float
        Minimum value to tick on the axis.

    max : int or float
        Maximum value to tick on the axis.

    number_of_ticks : int
        The number of tickmarks to plot. Must be > 2.

    show_zero : bool
        If set to ``True`` the zero tickmark will be plotted.

    Returns
    -------
    :class:`~numpy.ndarray`
        A 1d array with tickmark values.
    """
    range = max - min
    exponent = int(np.abs(np.round(np.log10(range))) + 1)
    magnitude = 10**exponent
    min = round(min, exponent)
    max = round(max, exponent)
    ticks, step = np.linspace(min, max, number_of_ticks, retstep=True)
    if show_zero and 0 not in ticks:
        positive_ticks = ticks[ticks >= 0]
        negative_ticks = ticks[ticks < 0]
        positive_ticks = np.insert(positive_ticks,0,0.0)
        ticks = np.hstack((negative_ticks, positive_ticks))
    return ticks


def set_matplotlib_rc_params():
    """
    Set matplotlib rcparams for plotting
    """
    font = {'family'        : 'sans-serif',
            'sans-serif'    : 'Helvetica',
            'style'         : 'normal',
            'variant'       : 'normal',
            'weight'        : 'medium',
            'stretch'       : 'normal',
            'size'          : 12.0}
    rc('font', **font)

    legend = {'fontsize'    : 10.0}
    rc('legend', **legend)

    axes = {'titlesize'     : 14.0,
            'labelsize'     : 12.0}
    rc('axes', **axes)
    rc('pdf', fonttype=42)

    ticks = {'direction'    : 'out',
             'labelsize'    : 12.0,
             'major.pad'    : 4,
             'major.size'   : 5,
             'major.width'  : 1.0,
             'minor.pad'    : 4,
             'minor.size'   : 2.5,
             'minor.width'  : 0.75}
    rc('xtick', **ticks)
    rc('ytick', **ticks)
