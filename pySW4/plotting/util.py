# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Plotting utilities
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------
"""
- utils.py -

Python module with general utilities for making plotting easier.

By: Shahar Shani-Kadmiel, August 2012, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

from matplotlib import rc
import numpy as np


def pretty_ticks(min, max, number_of_ticks, show_zero=False):
    """Function for controlling tickmarks.

    `show_zero` forces the zero tickmark to be plotted

    The rest of the parameters are self explanatory...
    """
    range  = max-min
    exponent = int(np.abs(np.round(np.log10(range)))+1)
    magnitude = 10**exponent
    min = round(min, exponent)
    max = round(max, exponent)
    ticks, step = np.linspace(min, max, number_of_ticks, retstep=True)
    if show_zero:
        positive_ticks = np.arange(0,max,step)
        negative_ticks = np.arange(0,min,-step)[::-1]
        ticks = np.hstack((negative_ticks[:-1],positive_ticks))
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
