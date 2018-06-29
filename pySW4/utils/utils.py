# -*- coding: utf-8 -*-
"""
Python module with general utilities.

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

import sys
from warnings import warn
from scipy.interpolate import griddata
import numpy as np
from pyproj import Proj, transform
try:
    import cv2
except ImportError:
    warn("OpenCV not found. Don't worry about this unless you want to use "
         "`pySW4.utils.resample` in `draft` mode.")
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
        a single 2d :class:`~numpy.ndarray` ``Z``. If a ``draft=True``
        and a single ``Z`` array is passed, ``corners`` must be
        supplied.

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
        arbitrary corners. See `Stack Overflow post
        <http://stackoverflow.com/a/31724678/1579211>`_ for explanation.

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
    """
    if len(data) == 3:
        X, Y, Z = data
    elif data.ndim == 2:
        X, Y, Z = None, None, data
    else:
        print('Error: data must be a tuple of 3x2d '
              ':class:`numpy.ndarry` (X, Y, Z) or one 2d ``Z`` array')

    w, e, s, n = extent

    # if no output shape is supplied
    # use the shape of the input array Z
    if not shape:
        shape = Z.shape
    z_ny, z_nx = Z.shape
    ny, nx = shape

    if X is None and Y is None:
        X, Y = np.meshgrid(np.linspace(w, e, z_nx),
                           np.linspace(s, n, z_ny))

    if draft is False:  # the long and accurate way...
        if verbose:
            message = ('Accurately interpolating data onto a grid of '
                       'shape %dx%d and extent %.2f,%.2f,%.2f,%.2f '
                       'using X, Y, Z arrays.\n'
                       '                      This may take a while...')
            print(message.format(ny, nx, w, e, s, n))
            sys.stdout.flush()

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
        List of `matplotlib named colors
        <http://matplotlib.org/examples/color/named_colors.html>`_.
        Arrange your colors so that the first color is the lowest value
        for the colorbar and the last is the highest.

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


def trim_cmap(cmap, cmin, cmax, n=256):
    """
    Trim a `cmap` to `cmin` and `cmax` limits in the range 0 to 1.
    Use :class:`~matplotlib.colors.Normalize` with `vmin`, `vmax` of
    the plot or data to get `cmin` and `cmax`

    Examples
    --------
    >>> norm = Normalize(-5, 10)
    >>> norm(-2)
    0.20000000000000001
    >>> norm(6)
    0.73333333333333328

    """
    cmap_ = LinearSegmentedColormap.from_list(
        'trim({},{:f},{:f})'.format(cmap.name, cmin, cmax),
        cmap(np.linspace(cmin, cmax, n)))
    return cmap_


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
    `x` and `y`.

    Returns
    -------
    2x list
        - one list of longitudes (or `x`) and
        - another list of latitudes (or `y`).
    """
    xc = [x[0, 0], x[0, -1], x[-1, -1], x[-1, 0]]
    yc = [y[0, 0], y[0, -1], y[-1, -1], y[-1, 0]]

    return xc, yc


def line_in_loglog(x, m, b):
    """
    Function to calculate the :math:`y` values of a line in log-log
    space according to

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
        `x` coordianates (might also be longitudes).

    y : float or sequence
        `y` coordianates (might also be latitudes).

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


def geographic2cartesian(proj4=None,
                         sw4_default_proj='+proj=latlong +datum=NAD83',
                         lon0=0, lat0=0, az=0,
                         m_per_lat=111319.5, m_per_lon=None,
                         lon=None, lat=None, x=None, y=None):
    """
    Calculate SW4 cartesian coordinats from geographic coordinates
    `lon`, `lat` in the `proj4` projection. If `x`, `y`
    coordinates are given (`x` and `y` are not ``None``) the
    inverse transformation is performed.

    Parameters
    ----------
    proj4 : str or dict or None
        EPSG or Proj4. If EPSG then string should be 'epsg:number'
        where number is the EPSG code. See the `Geodetic Parameter
        Dataset Registry <http://www.epsg-registry.org/>`_ for more
        information. If Proj4 then either a dictionary:
        ::

            {'proj': projection, 'datum': 'datum', 'units': units,
             'lon_0': lon0, 'lat_0': lat0, 'scale': scale ... etc.}

        or a string:
        ::

            '+proj=projection +datum=datum +units=units +lon_0=lon0 \\
                +lat_0=lat0 +scale=scale ... etc.'

        should be given.

        See the `Proj4 <https://trac.osgeo.org/proj/wiki/GenParms>`_
        documentation for a list of general Proj4 parameters.

        In the context of SW4 this dictionary or string should be taken
        from the ``grid`` line in the SW4 input file.

    sw4_default_proj : str
        SW4 uses '+proj=latlong +datum=NAD83' as the default projection.
        Pass a different projection for other purposes. See ``proj4``.

    lon0 : float
        Longitude of the grid origin in degrees.

        This is the ``lon`` keyword on the ``grid`` line in the SW4
        input file.

    lat0 : float
        Latitude of the grid origin in degrees.

        This is the ``lat`` keyword on the ``grid`` line in the SW4
        input file.

    az : float
        Azimuth between North and the ``x`` axis of the SW4 grid.

        This is the ``az`` keyword on the ``grid`` line in the SW4 input
        file.

    m_per_lat : float
        How many m to a degree latitude. (default: 111319.5 m)

        This is the ``mlat`` keyword on the ``grid`` line in the SW4
        input file.

    m_per_lon : float
        How many m to a degree longitude. If ``None``, ``m_per_lat`` is
        used).

        This is the ``mlon`` keyword on the ``grid`` line in the SW4
        input file.

    lon : float
        Longitude coordinates in degrees to convert (geo. ===> cart.).

    lat : float
        Latitude coordinates in degrees to convert (geo. ===> cart.).

    x : float
        ``x`` coordinates in meters to convert (cart. ===> geo.).

    y : float
        ``y`` coordinates in meters to convert (cart. ===> geo.).

    Returns
    -------
    Converted coordinates:
    - x, y if geo. ===> cart. was invoked or
    - lon, lat if cart. ===> geo. was invoked.

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

    """

    # No Proj4 projection, use simple transformations
    if not proj4 and lon is not None and lat is not None:
        x_out, y_out = simple_lonlat2xy(lon, lat, lon0, lat0, az,
                                        m_per_lat, m_per_lon)
    elif not proj4 and x is not None and y is not None:
        x_out, y_out = simple_xy2lonlat(x, y, lon0, lat0, az,
                                        m_per_lat, m_per_lon)

    # Proj4 projection is used, follow these tranformations
    else:
        # Proj4
        try:
            geo_proj = Proj(init=proj4)
        except RuntimeError:
            geo_proj = Proj(proj4)

        # sw4 projection (either the default or a user's choice)
        try:
            sw4_proj = Proj(init=sw4_default_proj)
        except RuntimeError:
            sw4_proj = Proj(sw4_default_proj)

        xoffset, yoffset = transform(sw4_proj, geo_proj, lon0, lat0)

        # lon, lat ===> x, y
        if lon is not None and lat is not None:
            x_, y_ = transform(sw4_proj, geo_proj, lon, lat)
            x_ -= xoffset
            y_ -= yoffset

            az_ = np.radians(az)
            x_out = x_ * np.sin(az_) + y_ * np.cos(az_)
            y_out = x_ * np.cos(az_) - y_ * np.sin(az_)

        # x, y ===> lon, lat
        elif x is not None and y is not None:
            az_ = np.radians(az)
            x_ = x * np.sin(az_) + y * np.cos(az_) + xoffset
            y_ = x * np.cos(az_) - y * np.sin(az_) + yoffset

            x_out, y_out = transform(geo_proj, sw4_proj, x_, y_)

    return x_out, y_out


def simple_xy2lonlat(x, y, lon0=0, lat0=0, az=0,
                     m_per_lat=111319.5, m_per_lon=None):
    """
    Project cartesian `x, y` coordinates to geographical `lat, lon`.

    This transformation is based on the default projection used by
    SW4. See section *3.1.1* of the `SW4
    <https://geodynamics.org/cig/software/sw4/>`_ User Guide for more
    details.

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
        `x` coordianates.

    y : float or sequence
        `y` coordianates.

    origin : 2-tuple or list
        `lat, lon` coordinates of the bottom-right corner of the
        `x, y` grid.

    az : float
        Rotation of the grid aroud the vertical (z-axis). See the
        SW4 User Guide for more information.

    m_per_lat : float
        How many m to a degree latitude. (default: 111319.5 m)

    m_per_lon : float
        How many m to a degree longitude. If ``None``, `m_per_lat` is
        used).

    Returns
    --------
    2x :class:`~numpy.ndarray`
        Two arrays with `lon` and `lat` projected points.
    """

    az_ = np.radians(az)
    lat = lat0 + (x * np.cos(az_) - y * np.sin(az_)) / m_per_lat

    if m_per_lon:
        lon = lon0 + (x * np.sin(az_) + y * np.cos(az_)) / m_per_lon
    else:
        lon = (lon0 + (x * np.sin(az_) + y * np.cos(az_)) /
               (m_per_lat * np.cos(np.radians(lat))))

    return lon, lat


def simple_lonlat2xy(lon, lat, lon0=0, lat0=0, az=0,
                     m_per_lat=111319.5, m_per_lon=None):
    """
    Project geographical `lat, lon` to cartesian `x, y` coordinates.

    This is the inverse function of :func:`~pySW4.utils.utils.xy2latlon`
    function.

    Parameters
    -----------
    lon : float or sequence
        `lon` coordianates.

    lat : float or sequence
        `lat` coordianates.

    origin : 2-tuple or list
        `lat, lon` coordinates of the bottom-right corner of the
        `x, y` grid.

    az : float
        Rotation of the grid aroud the vertical (z-axis). See the
        SW4 User Guide for more information.

    m_per_lat : float
        How many m to a degree latitude. (default: 111319.5 m)

    m_per_lon : float
        How many m to a degree longitude. If ``None``, `m_per_lat` is
        used).

    Returns
    --------
    2x :class:`~numpy.ndarray`
        Two arrays with ``x`` and ``y`` projected points.
    """

    az_ = np.radians(az)
    if m_per_lon:
        x = (m_per_lat * np.cos(az_) * (lat - lat0)
             + m_per_lon * (lon - lon0) * np.sin(az_))
        y = (m_per_lat * -np.sin(az_) * (lat - lat0)
             + m_per_lon * (lon - lon0) * np.cos(az_))
    else:
        x = (m_per_lat * (np.cos(az_) * (lat - lat0)
                          + (np.cos(np.radians(lat)) * (lon - lon0)
                             * np.sin(az_))))
        y = (m_per_lat * (-np.sin(az_) * (lat - lat0)
                          + np.cos(np.radians(lat)) * (lon - lon0)
                          * np.cos(az_)))

    return x, y


def _list(x):
    try:
        return list(x)
    except TypeError:
        return [x]


def nearest_values(array, value, threshold, retvalue=False):
    """
    Get the indices (or elements) of an array with a threshold range
    around a central value.

    If `retvalue` is true, the elements of the array are returned,
    otherwise a True or False array of the indices of these elements
    is returned.

    Examples
    --------
    >>> array = np.random.randint(-10, 10, size=10)
    >>> print(array)
    [ 3 -4  9  6  8 -1 -4 -7 -6  7]
    >>>
    >>> idx = nearest_values(array, 5, 2)
    >>> print(idx)
    >>> print(nearest_values(array, 5, 2, retvalue=True))
    >>> print array[idx]
    [ True False False  True False False False False False  True]
    [3 6 7]
    [3 6 7]
    """
    array = np.array(array)  # make sure aray is a numpy array
    left = value - threshold
    right = value + threshold
    idx = (array >= left) * (array <= right)
    if retvalue:
        return array[idx]
    return idx
