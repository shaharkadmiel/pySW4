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
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable


# set some matplotlib rcparams for plotting
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


# core plotting routine
def patch_plot(patch, ax=None, vmin='min', vmax='max', colorbar=True,
               **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_aspect(1)

        if vmax is 'max':
            vmax = patch.max
        elif type(vmax) is str:
            try:
                factor = float(vmax)
                vmax = factor*patch.rms
                if patch.min < 0:
                    vmin = -vmax

            except ValueError:
                print ('Warning! keyword vmax=$s in not understood...\n' %vmax,
                       'Setting to max')
                vmax = patch.max

        if vmin is 'min':
            vmin = patch.min

        if vmin > patch.min and vmax < patch.max:
            extend = 'both'
        elif vmin == patch.min and vmax == patch.max:
            extend = 'neither'
        elif vmin > patch.min:
            extend = 'min'
        else:# vmax < patch.max:
            extend = 'max'

        print vmin, vmax
        im = ax.imshow(patch.data.T, extent=patch.extent, vmin=vmin, vmax=vmax,
                       **kwargs)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im, cax=cax,
                              extend=extend,
                              label=colorbar if type(colorbar) is str else '')
        else:
            cb = None

        try:
            return fig, ax, cb
        except NameError:
            return cb