# -*- coding: utf-8 -*-
"""
Python module with general utilities.

.. module:: utils

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import sys
import warnings
from scipy.interpolate import griddata
import numpy as np
try:
    import cv2
except ImportError:
    warnings.warn('OpenCV not found, you will not be able to use '
                  '`pySW4.utils.resample` in `draft` mode.')
from matplotlib.colors import LinearSegmentedColormap, ColorConverter


def resample(data, extent, shape=None, method='linear',
             draft=False, corners=None, origin='nw', verbose=False):
    """
    Resample or interpolate an array on to a grid with new extent and or
    new shape.

    Parameters
    ----------
    data : 3-tuple or :class:`~numpy.ndarray`
        Either a tuple of 3x2d :class:`~numpy.ndarray` s ``(X,Y,Z)`` or
        a single 2d :class:`~numpy.ndarray` ``Z``. If a single ``Z``
        array is passed, ``corners`` must be passed.

    extent : tuple or list
        Extent of the output grid to which the array is interpolated.

    shape : tuple
        The shape of the output grid, defaults to the shape of the input
        array. If no shape is passed, output grid is the same shape as
        input array.

    method : str
        Method of interpolation. One of

        - ``nearest`` - return the value at the data point closest to
          the point of interpolation. This is the faster of the three.
        - ``linear`` - tesselate the input point set to n-dimensional
          simplices, and interpolate linearly on each simplex.
        - ``cubic`` - return the value determined from a piecewise
          cubic, continuously differentiable (C1), and approximately
          curvature-minimizing polynomial surface.

    draft : bool
        If set to True, a very fast OpenCV algorithem is used to
        interpolate the array by reprojection the data corners onto
        arbitrary corners. See `Stack Overflow post`_ for explanation.

    corners : 4-tuple
        Points to which the corners of ``Z`` will be tranformed to.
        After transformation, the TL, TR, BR, BL corners of the ``Z``
        array will be transformed to the coordinates in:

        ``corners=(top-left, top-right, bottom-right, bottom-left)``

        where each point is a tuple of ``(lon, lat)``.
        Optionally, corners can be a ``(src, dst)`` tuple, where
        ``src``, ``dst`` are each a tuple of ``n`` points. Each point in
        ``src`` is transformed to the respective point in extent
        coordinates in ``dst``.

    origin : str
        Corresponds to ``Z[0, 0]`` of the array or grid,
        either ``'nw'`` (default), or ``'sw'``.

    verbose : bool
        If set to True, some information about the transformation is
        provided.

    Returns
    -------
    3x2d :class:`~numpy.ndarray` s or 1x2d :class:`~numpy.ndarray`
        If ``data`` is a tuple of 3x2d arrays ``(X,Y,Z)`` and ``draft``
        is ``False``, ``xi, yi, zi`` are returned. If ``draft`` is
        ``True``, only ``zi`` is returned.


    .. _Stack Overflow post:
       http://stackoverflow.com/a/31724678/1579211
    """
    if len(data) == 3:
        X, Y, Z = data
    elif data.ndim == 2:
        Z = data
    else:
        print('Error: data must be 3x2d tuple of (XYZ) or 2d Z array')

    w, e, s, n = extent

    # if no output shape is supplied
    # use the shape of the input array Z
    if not shape:
        shape = Z.shape
    ny, nx = shape

    if draft is False:  # the long and accurate way...
        if verbose:
            message = ('Accurately interpolating data onto a grid of '
                       'shape %dx%d and extent %.2f,%.2f,%.2f,%.2f '
                       'using X, Y, Z arrays.\n'
                       '                      This may take a while...')
            print(message.format(ny, nx, w, e, s, n))
            sys.stdout.flush()
        nx, ny = shape
        xi, yi = np.meshgrid(np.linspace(w, e, nx),
                             np.linspace(s, n, ny))
        zi = griddata((X.ravel(), Y.ravel()), Z.ravel(), (xi, yi),
                      method=method)
        return xi, yi, zi

    elif draft is True:  # the fast and less accurate way...
        try:  # both src and dst are passed
            src, dst = corners
            src, dst = tuple(src), tuple(dst)

        except ValueError:  # only dst corners passed
            src = ((0, 0), (nx, 0), (nx, ny), (0, ny))
            dst = corners

        xc, yc = [p[0] for p in dst], [p[1] for p in dst]
        xc, yc = xy2pixel_coordinates(xc, yc, extent, shape, origin)
        dst = tuple(zip(xc, yc))

        if verbose:
            message = ('Transforming points:\n'
                       '%s\n'
                       'in the data to points:\n'
                       '%s\n'
                       'in the the output grid of shape %dx%d and '
                       'extent %.2f,%.2f,%.2f,%.2f.')
            print(message.format(src, dst, nx, ny, w, e, s, n))
            sys.stdout.flush()

        # Compute the transformation matrix which places
        # the corners Z at the corners points bounding the
        # data in output grid pixel coordinates
        tranformation_matrix = cv2.getPerspectiveTransform(np.float32(src),
                                                           np.float32(dst))

        # Make the transformation
        interpolation = {'nearest' : 0,
                         'linear'  : 1,
                         'cubic'   : 2}
        zi = cv2.warpPerspective(Z, tranformation_matrix,
                                 (nx, ny),
                                 flags=interpolation[method],
                                 borderMode=0,
                                 borderValue=np.nan)

        return zi


def make_cmap(colors, position=None, bit=False, named=True):
    """
    Make a color map compatible with matplotlib plotting functions.

    Parameters
    ----------
    colors : list
        List of `matplotlib named colors`_. Arrange your colors so that
        the first color is the lowest value for the colorbar and the
        last is the highest.

    position : list
        List of values from 0 to 1, controls the position of each color
        on the cmap. If None (default), an evenly spaced cmap is
        created.

        **Tip:**
            You can construct the positions sequence in data units and
            then normalize so that the forst value is 0 and the last
            value is 1.

    named : bool
        If set to False, colors are regarded as tuples of RGB values.
        The RGB values may either be in 8-bit [0 to 255], in which case
        ``bit`` must be set to True or arithmetic [0 to 1] (default).

    Returns
    -------
    :class:`~matplotlib.colors.LinearSegmentedColormap`
        An instance of the
        :class:`~matplotlib.colors.LinearSegmentedColormap` cmap class.


    .. _matplotlib named colors:
       http://matplotlib.org/examples/color/named_colors.html
    """
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            raise ValueError('Position length must be the same as colors\n')
        elif position[0] != 0 or position[-1] != 1:
            raise ValueError('Position must start with 0 and end with 1\n')
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red'   : [],
             'green' : [],
             'blue'  : []}
    if named:
        colors = [ColorConverter().to_rgb(c) for c in colors]
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


def flatten(ndarray):
    """
    Returns a flattened 1Darray from any multidimentional array. This
    function is recursive and may take a long time on large data.
    """
    for item in ndarray:
        try:
            for subitem in flatten(item):
                yield subitem
        except TypeError:
            yield item


def close_polygon(x, y):
    """
    Return a list of points with the first point added to the end.
    """
    return list(x) + [x[0]], list(y) + [y[0]]


def get_corners(x, y):
    """
    Extract the corners of the data from its coordinates arrays
    ``x`` and ``y``.

    Returns
    -------
    2x list
        - one list of longitudes (or ``x``) and
        - another list of latitudes (or ``y``).
    """
    xc = [x[0, 0], x[0, -1], x[-1, -1], x[-1, 0]]
    yc = [y[0, 0], y[0, -1], y[-1, -1], y[-1, 0]]

    return xc, yc


def line_in_loglog(x, m, b):
    """
    Function to calculate the ``y`` values of a line in log-log space
    according to

    .. math:: y(x) = b \cdot 10^{\log_{10} x^m},

    where:

    - :math:`m` is the slope of the line,
    - :math:`b` is the :math:`y(x=1)` value and
    - :math:`x` is a sequence of values on the x-axis.
    """
    return b * 10**(np.log10(x**m))


def xy2pixel_coordinates(x, y, extent, shape, origin='nw'):
    """
    Get the pixel coordinates on a grid.

    Parameters
    ----------
    x : float or sequence
        ``x`` coordianates (might also be longitudes).

    y : float or sequence
        ``y`` coordianates (might also be latitudes).

    extent : tuple or list
        Extent of the output grid to which the array is interpolated.

    shape : tuple
        The shape of the output grid, defaults to the shape of the input
        array. If no shape is passed, output grid is the same shape as
        input array.

    origin : str
        Corresponds to ``Z[0, 0]`` of the array or grid,
        either ``'nw'`` (default), or ``'sw'``.

    Returns
    -------
    2x list
        - one list of longitudes (or ``x``) and
        - another list of latitudes (or ``y``).
    """
    x, y = np.array([x, y])
    w, e, s, n = extent
    ny, nx = shape
    dx = float(e - w) / nx
    dy = float(n - s) / ny

    if origin is 'nw':
        xc = (x - w) / dx
        yc = (n - y) / dy
    elif origin is 'sw':
        xc = (x - w) / dx
        yc = (y - s) / dy
    else:
        raise ValueError("Origin must be 'nw' or 'sw'.")

    return xc, yc


def rms(x):
    """
    Returns the Root Mean Square of :class:`~numpy.ndarray` ``x``.
    """
    return np.sqrt((x**2).mean())


def calc_stuff(x):
    """
    Calculate ``min``, ``max``, ``rms``, and ``ptp`` on a list of
    2d :class:`~numpy.ndarray` ``x``.
    """
    xmax = []
    xmin = []
    xrms = []
    xptp = []
    for item in x:
        xmax += [item.max()]
        xmin += [item.min()]
        xrms += [rms(item)]
    xmax = max(xmax)
    xmin = min(xmin)
    xrms = rms(np.array(xrms))
    xptp = xmax - xmin
    return xmax, xmin, xrms, xptp


def xy2latlon(x, y, origin=(37.0, -118.0), az=0, km2deg=111.3195):
    """
    Project cartesian ``x, y`` coordinates to geographical ``lat, lon``.

    This transformation is based on the default projection used by
    `SW4`_. See section *3.1.1* of the `SW4`_ User Guide for more
    details.

    Note
    ----

    ::

             X
           ⋰
         ⋰
        o------->Y
        |
        |
        V
        Z

    If the azimuth of the SW4 grid is ``0`` this translates to:

    - X == Northing
    - Y == Easting
    - Z == Vertical (inverted!)


    Parameters
    ----------
    x : float or sequence
        ``x`` coordianates.

    y : float or sequence
        ``y`` coordianates.

    origin : 2-tuple or list
        ``lat, lon`` coordinates of the bottom-right corner of the
        ``x, y`` grid.

    az : float
        Rotation of the grid aroud the vertical (z-axis). See the
        `SW4`_ User Guide for more information.

    km2deg : float
        How many km to a degree. (default: 111.3195 km)

    Returns
    --------
    2x :class:`~numpy.ndarray`
        Two arrays with ``lat`` and ``lon`` projected points.


    .. _SW4:
       https://geodynamics.org/cig/software/sw4/
    """

    az = np.radians(az)
    lat = origin[0] + (x * np.cos(az) - y * np.sin(az)) / km2deg
    lon = (origin[1] + (x * np.sin(az) + y * np.cos(az)) /
           (km2deg * np.cos(np.radians(lat))))

    return lat, lon


def latlon2xy(lat, lon, origin=(37.0, -118.0), az=0, km2deg=111.3195):
    """
    Project geographical ``lat, lon`` to cartesian ``x, y`` coordinates.

    This is the inverse function of :func:`~pySW4.utils.utils.xy2latlon`
    function.

    Parameters
    -----------
    lat : float or sequence
        ``lat`` coordianates.

    lon : float or sequence
        ``lon`` coordianates.

    origin : 2-tuple or list
        ``lat, lon`` coordinates of the bottom-right corner of the
        ``x, y`` grid.

    az : float
        Rotation of the grid aroud the vertical (z-axis). See the
        `SW4`_ User Guide for more information.

    km2deg : float
        How many km to a degree. (default: 111.3195 km)

    Returns
    --------
    2x :class:`~numpy.ndarray`
        Two arrays with ``x`` and ``y`` projected points.
    """

    az = np.radians(az)
    x = km2deg * (
        ((lat - origin[0]) +
         (lon - origin[1]) * np.cos(np.radians(lat)) * np.tan(az)) /
        (np.cos(az) * (1 + np.tan(az)**2)))
    y = (km2deg * ((lon - origin[1]) * np.cos(np.radians(lat))) /
         np.cos(az)) - x * np.tan(az)

    return x, y
