# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Plotting routines for plotting WPP and SW4 images
#   Author: Shahar Shani-Kadmiel
#           kadmiel@post.bgu.ac.il
#
# Copyright ©(C) 2012-2014 Shahar Shani-Kadmiel
# This code is distributed under the terms of the GNU General Public License
# -----------------------------------------------------------------------------

"""
- image_plot.py -

Python module with plotting routines for plotting WPP and SW4 images.

By: Shahar Shani-Kadmiel, September 2015, kadmiel@post.bgu.ac.il

"""
from matplotlib import rc


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

# leaving this stub only for backwards compatibility right now
def patch_plot(patch, *args, **kwargs):
    """
    """
    patch.plot(*args, **kwargs)
