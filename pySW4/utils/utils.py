"""
- utils.py -

Python module with general utilities.

By: Shahar Shani-Kadmiel, August 2012, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

import sys, warnings
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
    """Resample or interpolate an array on to a grid with new extent.
    If no shape is passed, output grid is same shape as input array.

    Params:
    -------

    data : either a tuple of 3x2d arrays `(X,Y,Z)` or a single
        2d array `Z`. If a single `Z` array is passed, `corners` must
        be passed.

    extent : tuple/list the extent of the output grid to which the
        array is interpolated.

    shape : the shape of the output grid, tuple, defaults to the shape
        of the input array.

    method : 'linear' interpolation (default), 'nearest or 'cubic'
        interpolation are also possible.

    draft : if set to True, a very fast OpenCV algorithem is used to
        interpolate the array by reprojection the data corners onto
        arbitrary corners. See http://stackoverflow.com/a/31724678/1579211
        for explanation.

    corners : a tuple of 4 points to which the corners of `Z` will
        be tranformed to.
        After transformation, the TL,TR,BR,BL corners of the `Z` array
        will be transformed to the coordinates in:
        `corners=(top-left, top-right, bottom-right, bottom-left)` where
        each point is a tuple of `(lon,lat)`.
        Optionally, corners can be a `(src,dst)` tuple, where src,dst are
        each a tuple of n points. Each point in src is transformed to the
        respective point in extent coordinates in dst.

    origin : corresponds to Z[0,0] of the array or grid,
        either 'nw' (default), or 'sw'.

    verbose : if set to True, some information about the transformation is
        provided.

    Returns :
    ---------
    If `data` is a tuple of 3x2d arrays `(X,Y,Z)` and `draft` is False,
    xi, yi, zi are returned. If `draft` is True, only zi is returned
    """

    if len(data) == 3:
        X,Y,Z = data
    elif data.ndim == 2:
        Z = data
    else:
        print('Error: data must be 3x2d tuple of (XYZ) or 2d Z array')

    w,e,s,n = extent

    # if no output shape is supplied
    # use the shape of the input array Z
    if not shape:
        shape = Z.shape
    ny,nx = shape

    if draft is False: # the long and accurate way...
        if verbose:
            message = """
Accurately interpolating data onto a grid of shape
%dx%d and extent %.2f,%.2f,%.2f,%.2f using X,Y,Z arrays.
This may take a while...
            """
            print(message %(ny,nx,w,e,s,n))
            sys.stdout.flush()
        nx,ny = shape
        xi,yi = np.meshgrid(np.linspace(w, e, nx),
                            np.linspace(s, n, ny))
        zi = griddata((X.ravel(),Y.ravel()), Z.ravel(), (xi,yi), method=method)
        return xi, yi, zi

    elif draft is True: # the fast and less accurate way...
        try: # both src and dst are passed
            src,dst = corners
            src,dst = tuple(src),tuple(dst)

        except ValueError: # only dst corners passed
            src = ((0,0),(nx,0),(nx,ny),(0,ny))
            dst = corners

        xc,yc = [p[0] for p in dst],[p[1] for p in dst]
        xc,yc = xy2pixel_coordinates(xc,yc,extent,shape,origin)
        dst = tuple(zip(xc,yc))

        if verbose:
            message = """
Transforming points:
%s
in the data to points:
%s
in the the output grid of shape %dx%d and extent %.2f,%.2f,%.2f,%.2f.
            """
            print(message %(src,dst,nx,ny,w,e,s,n))
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
        zi = cv2.warpPerspective(Z,tranformation_matrix,
                                 (nx,ny),
                                 flags=interpolation[method],
                                 borderMode=0,
                                 borderValue=np.nan)

        return zi


def make_cmap(colors, position=None, bit=False, named=True):
    """Make a color map compatible with matplotlib plotting functions.

    Params :
    --------

    colors : a list of matplotlib named colors, see
        http://matplotlib.org/examples/color/named_colors.html
        for color names. Arrange your colors so that the first color is
        the lowest value for the colorbar and the last is the highest.

    position : a list of values from 0 to 1, controls the position of
        each color on the cmap. If None (default), an evenly spaced cmap
        is created.

    named : if set to False, colors are regarded as tuples of RGB values.
        The RGB values may either be in 8-bit [0 to 255], in which case
        bit must be set to True or arithmetic [0 to 1] (default).

    Returns :
    ---------
    a cmap (an instance of `matplotlib.colors.LinearSegmentedColormap`)
    """

    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            return "Error! position length must be the same as colors\n"
        elif position[0] != 0 or position[-1] != 1:
            return "Error! position must start with 0 and end with 1\n"
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    if named:
        colors = [ColorConverter().to_rgb(c) for c in colors]
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def flatten(ndarray):

    """Returns a flattened 1Darray from any multidimentional
    array. This function is recursive and may take a long time
    on large data"""

    for item in ndarray:
        try:
            for subitem in flatten(item):
                yield subitem
        except TypeError:
            yield item


def close_polygon(x,y):
    """Return a list of points with the first point added to the end"""
    return list(x) + [x[0]], list(y) + [y[0]]


def get_corners(x,y):
    """extract the corners of the data from
    its coordinates arrays x and y.

    returns 2 lists of longitudes and latitudes"""

    xc = [x[0,0],x[0,-1],x[-1,-1],x[-1,0]]
    yc = [y[0,0],y[0,-1],y[-1,-1],y[-1,0]]

    return xc,yc


def line_in_loglog(x, m, b):
    """
    Function to calculate the y values of a line
    in log-log space according to b * 10**(log10(x**m))
    where ``m`` is the slope of the line, ``b`` is the y
    value at x=1 and x is a sequence of values on the x-axis.
    """
    return b * 10**(np.log10(x**m))


def xy2pixel_coordinates(x, y, extent, shape, origin='nw'):
    """get the pixel coordinates of xy points (can be lon,lat) on
    a grid with `extent` and `shape`.
    `origin` corresponds to Z[0,0] of the array or grid, either
    'nw' (default), or 'sw'."""

    x,y = np.array([x,y])
    w,e,s,n = extent
    ny,nx = shape
    dx = float(e-w)/nx
    dy = float(n-s)/ny

    if origin is 'nw':
        xc = (x - w)/dx
        yc = (n - y)/dy
    elif origin is 'sw':
        xc = (x - w)/dx
        yc = (y - s)/dy
    else:
        print("Error: origin must be 'nw' or 'sw'.")

    return xc,yc


def rms(x):
    """Returns the Root Mean Square of numpy array"""
    return np.sqrt((x**2).mean())


def calc_stuff(x):
    """Calculate min,max,rms, and ptp on a list of 2darrays"""

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
    """Project cartesian x,y coordinates to geographical lat,lon.

    Parameters
    -----------
    x and y : 2darrays in km.

    origin : tuple or list, optional, default: lat=37.0 lon=-118.0
            lat,lon coordinates of the south-west corner of the x,y grid

    az : float or int, optional, default: 0 degrees
            rotation of the grid aroud the vertical (z-axis),
            see the SW4 User Guide for more information.

    km2deg : float, optional, default: 111.3195 km
            how many km to a degree.

    Returns
    --------
    lat and lon 2darrays."""

    az = np.radians(az)
    lat = origin[0] + (x * np.cos(az) - y * np.sin(az)) / km2deg
    lon = (origin[1] + (x * np.sin(az) + y * np.cos(az)) /
           (km2deg * np.cos(np.radians(lat))))

    return lat, lon


def latlon2xy(lat, lon, origin=(37.0, -118.0), az=0, km2deg=111.3195):
    """Project cartesian x,y coordinates to geographical lat,lon.

    Parameters
    -----------
    lat and lon : 2darrays in decimal degrees.

    origin : tuple or list, optional, default: lat=37.0 lon=-118.0
            lat,lon coordinates of the south-west corner of the x,y grid

    az : float or int, optional, default: 0 degrees
            rotation of the grid aroud the vertical (z-axis),
            see the SW4 User Guide for more information.

    km2deg : float, optional, default: 111.3195 km
            how many km to a degree.

    Returns
    --------
    x and y 2darrays."""

    az = np.radians(az)
    x = km2deg * (
        ((lat - origin[0]) +
         (lon - origin[1]) * np.cos(np.radians(lat)) * np.tan(az)) /
        (np.cos(az) * (1 + np.tan(az)**2)))
    y = (km2deg * ((lon - origin[1]) * np.cos(np.radians(lat))) /
         np.cos(az)) - x * np.tan(az)

    return x, y
