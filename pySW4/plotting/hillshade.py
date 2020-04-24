# -*- coding: utf-8 -*-
"""
Plotting routines for shaded-relief and GMT style draping of data.

.. module:: hillshade

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, rgb_to_hsv, hsv_to_rgb
# from matplotlib.colorbar import ColorbarBase
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import uniform_filter

INT_AND_FLOAT = ([int, float]
                 + np.sctypes.get('int')
                 + np.sctypes.get('uint')
                 + np.sctypes.get('float'))


def calc_intensity(relief, azimuth=315., altitude=45.,
                   scale=None, smooth=None, normalize=False):
    """
    Calculate the illumination intensity of ``relief``.

    Can be used as to create a shaded relief map and GMT style draping
    of data.

    It is assumed that the grid origin is at the upper-left corner.
    If that is not the case, add 90 to ``azimuth``.

    This function produces similar results to the
    :meth:`~matplotlib.colors.LightSource.hillshade` method of
    matplotlib but gives extra control in terms of how the result is
    normalized.

    Parameters
    ----------
    relief : a 2d :class:`~numpy.ndarray`
        Topography or other data to calculate intensity from.

    azimuth : float
        Direction of light source, degrees from north.

    altitude : float
        Height of light source, degrees above the horizon.

    scale : float
        Scaling value of the data.

    smooth : float
        Number of cells to average before intensity calculation.

    normalize : bool or float
        By default the intensity is clipped to the [0,1] range. If set
        to ``True``, intensity is normalized to [0,1]. Otherwise, give a
        float value to normalize to [0,1] and multiply by the value
        before clipping to [0,1]. If ``normalize`` > 1, illumination
        becomes brighter and if < 1 illumination becomes darker.

    Returns
    -------
    intensity : :class:`~numpy.ndarray`
        a 2d array with illumination in the [0,1] range.
        Same size as ``relief``.
    """

    relief = relief.copy()

    if scale is not None:
        relief = relief * scale
    if smooth:
        relief = uniform_filter(relief, size=smooth)

    dzdy, dzdx = np.gradient(relief)

    slope = 0.5 * np.pi - np.arctan(np.sqrt(dzdx**2 + dzdy**2))

    aspect = np.arctan2(dzdx, dzdy)

    altitude = np.radians(altitude)
    azimuth = np.radians((azimuth - 90) % 360)

    intensity = (np.sin(altitude) * np.sin(slope) +
                 np.cos(altitude) * np.cos(slope) *
                 np.cos(-azimuth - 0.5 * np.pi - aspect))

    if normalize:
        intensity = (normalize *
                     (intensity - intensity.min()) / intensity.ptp())

    return intensity.clip(0, 1)


def shade_and_color(relief, data=None, az=315, alt=45, vmax='max', vmin='min',
                    cmap='gist_earth', smooth=None, scale=None,
                    blend_mode='multiply', contrast=1, brightness=1,
                    method='matplotlib', normalize=False,
                    return_shading_dict=False, **kwargs):
    """
    Shade (relief) and color (relief or data).

    This is done using the :meth:`~matplotlib.colors.LightSource.shade`
    method of the :class:`~matplotlib.colors.LightSource` class.

    Parameters
    ----------
    relief : 2d :class:`~numpy.ndarray`
        Contains elevation (usually) data. Used for calculating
        intensities.

    data : None or 2d :class:`~numpy.ndarray`
        Data to color (drape) over the intensity grid. If ``None``,
        relief is colored over the intensity grid

    az : int or float
        The azimuth (0-360, degrees clockwise from North) of the light
        source. Defaults to 315 degrees (from the northwest).

    alt : int or float
        The altitude (0-90, degrees up from horizontal) of the light
        source. Defaults to 45 degrees from horizontal.

    vmax, vmin : str or float
        Used to clip the coloring of the data at the set value.
        Default is 'max' and 'min' which used the extent of the data.
        If  ``float``s, the colorscale saturates at the given values.
        Finally, if a string is passed (other than 'max' or 'min'), it
        is casted to float and used as an ``rms`` multiplier. For
        instance, if ``vmax='3'``, clipping is done at 3.0\*rms of the
        data.

        To force symmetric coloring around 0 set `vmin` to ``None`` or
        ``False``. This will cause `vmin` to equal -1 * `vmax`.

    cmap : str or :class:`~matplotlib.colors.Colormap` instance
        String of the name of the colormap, i.e. 'Greys', or a
        :class:`~matplotlib.colors.Colormap` instance.

    smooth : float
        Smooth the relief before calculating the intensity grid. This
        reduces noise. The overlaid relief is untouched.

    scale : float
        Scaling value of the data. Higher definition is achieved by
        scaling the elevation prior to the intensity calculation.
        This can be used either to correct for differences in units
        between the x, y coordinate system and the elevation coordinate
        system (e.g. decimal degrees vs meters).

    blend_mode : str or callable
        The type of blending used to combine the colormapped data values
        with the illumination intensity. For backwards compatibility,
        this defaults to 'hsv'. Note that for most topographic
        surfaces, 'overlay' or 'multiply' appear more visually
        realistic. If a user-defined function is supplied, it is
        expected to combine an MxNx3 RGB array of floats (ranging 0 to
        1) with an MxNx1 hillshade array (also 0 to 1). (Call signature
        func(rgb, illum, \*\*kwargs))

        Options are:
        **'hsv', 'overlay', 'soft'** - are achieved using the
        :meth:`~matplotlib.colors.LightSource.shade_rgb` method of the
        :class:`~matplotlib.colors.LightSource` class.

        **'multiply', 'hard', 'screen', 'pegtop'** - are achieved by
        image manipulation in RGB space. See :func:`~.multiply`,
        :func:`~.hard`, :func:`~.screen`, :func:`~.pegtop`.

    contrast : float
        Increases or decreases the contrast of the resulting image.
        If > 1 intermediate values move closer to full illumination or
        shadow (and clipping any values that move beyond 0 or 1).

    brightness : float
        Increases or decreases the brightness of the resulting image.
        Ignored for 'hsv', 'overlay', 'soft'.

    method : {'matplotlib', 'calc_intensity'}
        By default, matplotlib's
        :meth:`~matplotlib.colors.LightSource.hillshade` is used to
        calculate the illumination intensity of the relief.
        For better control, :func:`~.calc_intensity` can be used
        instead.

    normalize : bool or float
        By default the intensity is clipped to the [0,1] range. If set
        to ``True``, intensity is normalized to [0,1]. Otherwise, give a
        float value to normalize to [0,1] and then divide by the value.

        This is ignored if ``method`` is 'matplotlib'.

    return_shading_dict : bool
        False by default. If ``True``, a dictionary is returned that can
        be passed to :func:`~matplotlib.pyplot.imshow` and
        :func:`~.shade_colorbar` to preserve consistency.

    .. note:: It is assumed that relief and data have the same extent,
       origin and shape. Any resizing and reshaping must be done prior
       to this.

    Other parameters
    ----------------
    kwargs : ``blend_mode`` options.
    """
    if blend_mode in BLEND_MODES or callable(blend_mode):
        blend_func = BLEND_MODES[blend_mode]
    else:
        raise ValueError('"blend_mode" must be callable or one of {}'
                         .format(BLEND_MODES.keys))

    relief = relief.copy()
    data = data if data is not None else relief

    if scale is not None:
        relief = relief * scale
    if smooth:
        relief = uniform_filter(relief, size=smooth)

    # data
    if vmax is 'max':
        clip = np.nanmax(data)
    elif type(vmax) in INT_AND_FLOAT:
        clip = vmax
    else:
        clip = float(vmax) * np.nanstd(data)

    if vmin is 'min':
        vmin = np.nanmin(data)
    elif type(vmin) in INT_AND_FLOAT:
        pass
    elif vmin in [None, False]:
        vmin = -clip

    vmax = clip

    # matplotlib shade_rgb using hillshade intensity calculation
    if method == 'matplotlib':
        ls = LightSource(azdeg=az, altdeg=alt)

        # color and drape the data on top of the shaded relief
        rgb = ls.shade_rgb(
            data2rgb(data, cmap, vmin, vmax), relief,
            blend_mode=blend_func, fraction=contrast, **kwargs)

    # use calc_intensity with better control
    else:
        rgb = data2rgb(data, cmap, vmin, vmax)
        illumination = calc_intensity(relief, az, alt, normalize=normalize)

        illumination = illumination[..., np.newaxis]

        if blend_mode in BLEND_MODES:
            blend_func = BLEND_MODES[blend_mode]
        elif callable(blend_mode):
            blend_func = blend_mode
        else:
            raise ValueError('"blend_mode" must be callable or one of {}'
                             .format(BLEND_MODES.keys))

        rgb = blend_func(rgb, illumination)

    if brightness != 1:
        rgb = adjust_brightness(rgb, brightness)

    if return_shading_dict:
        return rgb, dict(cmap=cmap, vmin=vmin, vmax=vmax,
                         blend_mode=blend_func, brightness=brightness)
    return rgb


def data2rgb(data, cmap='Greys_r', vmin=None, vmax=None, norm=None,
             bytes=False):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    norm = norm or Normalize(vmin, vmax)
    return cmap(norm(data), bytes=bytes)[:, :, :3]


def adjust_brightness(rgb, brightness):
    hsv = rgb_to_hsv(rgb[:, :, :3])
    hsv[:, :, -1] *= brightness
    return hsv_to_rgb(hsv)


def multiply(rgb, illum):
    rgb = rgb[:, :, :3]
    illum = illum[:, :, :3]
    return np.clip(rgb * illum, 0.0, 1.0)


def screen(rgb, illum):
    rgb = rgb[:, :, :3]
    illum = illum[:, :, :3]
    return np.clip(1 - (1 - rgb) * (1 - illum), 0.0, 1.0)


def hard(rgb, illum):
    rgb = rgb[:, :, :3]
    illum = illum[:, :, :3]
    image = np.where(illum < 0.5,
                     2 * rgb * illum,
                     1 - 2 * (1 - illum) * (1 - rgb))
    return np.clip(image, 0.0, 1.0)


def pegtop(rgb, illum):
    rgb = rgb[:, :, :3]
    illum = illum[:, :, :3]
    image = (1 - 2 * rgb) * illum**2 + 2 * rgb * illum
    return np.clip(image, 0.0, 1.0)


def shade_colorbar(cb, max_illum=1, min_illum=0.1, n=3,
                   **kwargs):
    """
    Shade a colorbar to add illumination effects similar to GMT psscale.

    Parameters
    ----------
    cb : :class:`~matplotlib.colorbar.Colorbar`
        A colorbar instance to add illumination effects to.

    max_illum, min_illum : float
        The maximum (light) and minimum(dark) illumination values.

    n : int
        Number of illumination levels. A higher value generates more
        levels of illumination between ``min_illum`` and ``max_illum``
        but the result is linearly interpolated so 3 is a good enough.
        Larger values are slower.

    Other parameters
    ----------------
    kwargs : dict
        A dictionary returned by :func:`~.shade_and_color`. Expected
        keys are 'cmap', 'vmin', 'vmax', 'blend_mode', and 'brightness'.
        Other keywards are passed to blend_mode.
    """
    x = np.linspace(0, 1, 256)
    y = np.linspace(min_illum, max_illum, n)
    C1, C0 = np.meshgrid(x, y)

    xmin = ymin = cb.vmin
    xmax = ymax = cb.vmax

    if cb.orientation == 'vertical':
        C1 = C1.T[::-1]
        C0 = C0.T[:, ::-1]

    illumination = C0[::-1]**0.5

    cmap = cb.cmap
    rgb = data2rgb(C1, cmap)

    illumination = illumination[..., np.newaxis]

    blend_mode = kwargs['blend_mode']

    if blend_mode in BLEND_MODES:
        blend_func = BLEND_MODES[blend_mode]
    elif callable(blend_mode):
        blend_func = blend_mode
    else:
        raise ValueError('"blend_mode" must be callable or one of {}'
                         .format(BLEND_MODES.keys))

    rgb = blend_func(rgb, illumination)
    if kwargs['brightness'] != 1:
        rgb = adjust_brightness(rgb, kwargs['brightness'])
    cb.ax.imshow(rgb, interpolation='bilinear', aspect=cb.ax.get_aspect(),
                 extent=(xmin, xmax, ymin, ymax), zorder=1)


ls = LightSource()

# Blend the hillshade and rgb data using the specified mode
BLEND_MODES = {
    'hsv'      : ls.blend_hsv,
    'soft'     : ls.blend_soft_light,
    'overlay'  : ls.blend_overlay,
    'multiply' : multiply,
    'hard'     : hard,
    'screen'   : screen,
    'pegtop'   : pegtop,
}
