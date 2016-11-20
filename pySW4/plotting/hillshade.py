# -*- coding: utf-8 -*-
"""
Plotting routines for shaded-relief and draping data over shaded relief.

.. module:: hillshade

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import resample

INT_AND_FLOAT = ([int, float]
                 + np.sctypes.get('int')
                 + np.sctypes.get('uint')
                 + np.sctypes.get('float'))


def drape_plot(data, relief, extent, vmax='max', vmin='min',
               az=315, alt=45, cmap='hot_r', blend_mode='hsv',
               contrast=1, definition=1, origin='upper', ax=None,
               colorbar=True):
    """
    Drape `data` over `relief`.

    This is done using the
    :meth:`~matplotlib.colors.LightSource.shade_rgb` method of the
    :class:`~matplotlib.colors.LightSource` class.

    Parameters
    ----------
    data : a 2d :class:`~numpy.ndarray`
        Contains data to be draped over relief.

    relief : a 2d :class:`~numpy.ndarray`
        Contains elevation (usually).

    extent : array-like
        Extent of the domain plotted.
        (xmin,xmax,ymin,ymax) or (w,e,s,n)

    vmax, vmin : str or float
        Used to clip the coloring of the data at the set value.
        Default is 'max' and 'min' which used the extent of the data.
        If  ``float``s, the colorscale saturates at the given values.
        Finally, if a string is passed (other than 'max' or 'min'), it
        is casted to float and used as an ``rms`` multiplier. For
        instance, if ``vmax='3'``, clipping is done at 3.0\*rms of the
        data.

        To force symmetric coloring around 0 set `vmin` to ``None`` or
        ``False``. This will cause `vmin` to equal `vmax`.

    az : int or float
        The azimuth (0-360, degrees clockwise from North) of the light
        source. Defaults to 315 degrees (from the northwest).

    alt : int or float
        The altitude (0-90, degrees up from horizontal) of the light
        source. Defaults to 45 degrees from horizontal.

    cmap : str or :class:`~matplotlib.colors.Colormap` instance
        String of the name of the colormap, i.e. 'Greys', or a
        :class:`~matplotlib.colors.Colormap` instance used for the
        data.

    blend_mode : {'hsv', 'overlay', 'soft', callable}
        The type of blending used to combine the colormapped data values
        with the illumination intensity. For backwards compatibility,
        this defaults to 'hsv'. Note that for most topographic
        surfaces, 'overlay' or 'soft' appear more visually
        realistic. If a user-defined function is supplied, it is
        expected to combine an MxNx3 RGB array of floats (ranging 0 to
        1) with an MxNx1 hillshade array (also 0 to 1). (Call signature
        func(rgb, illum, \*\*kwargs))

    contrast : int or float
        Increases or decreases the contrast of the hillshade.
        If > 1 intermediate values move closer to full illumination or
        shadow (and clipping any values that move beyond 0 or 1).
        Note that this is not visually or mathematically the same as
        definition.

    definition : int or float
        Higher definition is achieved by scaling the elevation prior to
        the illumination calculation. This can be used either to correct
        for differences in units between the x,y coordinate system and
        the elevation coordinate system (e.g. decimal degrees vs meters)
        or to exaggerate or de-emphasize topography.

    origin : {'upper', 'lower'}
        Places the origin at the 'upper' (default) or 'lower' left
        corner of the plot.

    ax : :class:`~matplotlib.axes.Axes` instance
        Plot to an existing axes. If None, a
        :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes` are created and their instances
        are returned for further manipulation.

    colorbar : bool
        By default, a colorbar is drawn with the plot. If `colorbar`
        is a string, it is used for the label of the colorbar.
        Otherwise, the colorbar can be omitted by setting to False.
        :class:`~matplotlib.colorbar.Colorbar` instance is returned.

    .. note:: *All* grids must have the same extent and origin!
    """

    if not ax:
        fig, ax = plt.subplots()
        ax.axis(extent)
        ax.set_aspect(1)

    # data
    if vmax is 'max':
        clip = np.nanmax(data)
    elif type(vmax) in INT_AND_FLOAT:
        clip = vmax
    else:
        clip = float(vmax) * np.nanstd(data)

    if vmin is 'min':
        vmin = np.nanmin(data)
    elif type(vmin) in [int, float]:
        pass
    elif vmin in [None, False]:
        vmin = -clip

    vmax = clip

    if vmin > data.min() and vmax < data.max():
        extend = 'both'
    elif vmin > data.min():
        extend = 'min'
    elif vmax < data.max():
        extend = 'max'
    else:
        extend = 'neither'

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    im.remove()

    if data.size < relief.size:
        _, _, data = resample(data, extent, relief.shape)
    elif data.size > relief.size:
        _, _, relief = resample(relief, extent, data.shape)

    norm = Normalize(vmin, vmax)
    ls = LightSource(azdeg=az, altdeg=alt)
    if type(cmap) is str:
        cmap = plt.cm.get_cmap(cmap)

    rgba = ls.shade_rgb(cmap(norm(data)), relief, fraction=contrast,
                        blend_mode=blend_mode, vert_exag=definition)
    ax.imshow(rgba, extent=extent, origin=origin)
    # make sure no offsets are introduced
    ax.ticklabel_format(useOffset=False)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cb = plt.colorbar(im, cax=cax, extend=extend)
        if type(colorbar) is str:
            cb.set_label(colorbar)
        cb.solids.set_edgecolor('face')
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((-1, 4))
        cb.update_ticks()
    else:
        cb = None

    try:
        return fig, ax, cb
    except NameError:
        return cb


def hillshade_plot(relief, extent, vmax='max', vmin='min',
                   az=315, alt=45, cmap='gist_earth', blend_mode='hsv',
                   contrast=1, definition=1, origin='upper', ax=None,
                   colorbar=True):
    """
    Plot hillshade of `relief`.

    This is done using the :meth:`~matplotlib.colors.LightSource.shade`
    method of the :class:`~matplotlib.colors.LightSource` class.

    Parameters
    ----------
    relief : a 2d :class:`~numpy.ndarray`
        Contains elevation (usually).

    extent : list or tuple
        Extent of the domain plotted.
        (xmin,xmax,ymin,ymax) or (w,e,s,n)

    vmax, vmin : str or float
        Used to clip the coloring of the data at the set value.
        Default is 'max' and 'min' which used the extent of the data.
        If  ``float``s, the colorscale saturates at the given values.
        Finally, if a string is passed (other than 'max' or 'min'), it
        is casted to float and used as an ``rms`` multiplier. For
        instance, if ``vmax='3'``, clipping is done at 3.0\*rms of the
        data.

        To force symmetric coloring around 0 set `vmin` to ``None`` or
        ``False``. This will cause `vmin` to equal `vmax`.

    az : int or float
        The azimuth (0-360, degrees clockwise from North) of the light
        source. Defaults to 315 degrees (from the northwest).

    alt : int or float
        The altitude (0-90, degrees up from horizontal) of the light
        source. Defaults to 45 degrees from horizontal.

    cmap : str or :class:`~matplotlib.colors.Colormap` instance
        String of the name of the colormap, i.e. 'Greys', or a
        :class:`~matplotlib.colors.Colormap` instance.

    blend_mode : {'hsv', 'overlay', 'soft', callable}
        The type of blending used to combine the colormapped data values
        with the illumination intensity. For backwards compatibility,
        this defaults to 'hsv'. Note that for most topographic
        surfaces, 'overlay' or 'soft' appear more visually
        realistic. If a user-defined function is supplied, it is
        expected to combine an MxNx3 RGB array of floats (ranging 0 to
        1) with an MxNx1 hillshade array (also 0 to 1). (Call signature
        func(rgb, illum, \*\*kwargs))

    contrast : int or float
        Increases or decreases the contrast of the hillshade.
        If > 1 intermediate values move closer to full illumination or
        shadow (and clipping any values that move beyond 0 or 1).
        Note that this is not visually or mathematically the same as
        definition.

    definition : int or float
        Higher definition is achieved by scaling the elevation prior to
        the illumination calculation. This can be used either to correct
        for differences in units between the x,y coordinate system and
        the elevation coordinate system (e.g. decimal degrees vs meters)
        or to exaggerate or de-emphasize topography.

    origin : {'upper', 'lower'}
        Places the origin at the 'upper' (default) or 'lower' left
        corner of the plot.

    ax : :class:`~matplotlib.axes.Axes` instance
        Plot to an existing axes. If None, a
        :class:`~matplotlib.figure.Figure` and
        :class:`~matplotlib.axes.Axes` are created and their instances
        are returned for further manipulation.

    colorbar : bool
        By default, a colorbar is drawn with the plot. If `colorbar`
        is a string, it is used for the label of the colorbar.
        Otherwise, the colorbar can be omitted by setting to False.
        :class:`~matplotlib.colorbar.Colorbar` instance is returned.

    .. note:: *All* grids must have the same extent and origin!
    """

    if not ax:
        fig, ax = plt.subplots()
        ax.axis(extent)

    # data
    if vmax is 'max':
        clip = np.nanmax(relief)
    elif type(vmax) in INT_AND_FLOAT:
        clip = vmax
    else:
        clip = float(vmax) * np.nanstd(relief)

    if vmin is 'min':
        vmin = np.nanmin(relief)
    elif type(vmin) in INT_AND_FLOAT:
        pass
    elif vmin in [None, False]:
        vmin = -clip

    vmax = clip

    if vmin > relief.min() and vmax < relief.max():
        extend = 'both'
    elif vmin > relief.min():
        extend = 'min'
    elif vmax < relief.max():
        extend = 'max'
    else:
        extend = 'neither'

    im = ax.imshow(relief, cmap=cmap, vmin=vmin, vmax=vmax)
    im.remove()

    ls = LightSource(azdeg=az, altdeg=alt)
    if type(cmap) is str:
        cmap = plt.cm.get_cmap(cmap)

    rgba = ls.shade(relief, cmap=cmap, blend_mode=blend_mode,
                    vmin=vmin, vmax=vmax, vert_exag=definition,
                    fraction=contrast)
    ax.imshow(rgba, extent=extent, origin=origin)
    # make sure no offsets are introduced
    ax.ticklabel_format(useOffset=False)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cb = plt.colorbar(im, cax=cax, extend=extend)
        if type(colorbar) is str:
            cb.set_label(colorbar)
        cb.solids.set_edgecolor('face')
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((-1, 4))
        cb.update_ticks()
    else:
        cb = None

    try:
        return fig, ax, cb
    except NameError:
        return cb
