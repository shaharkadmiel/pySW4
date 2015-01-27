"""
- utils.py -

Python module with general utilities.

By: Shahar Shani-Kadmiel, August 2012, kadmiel@post.bgu.ac.il

"""
import numpy as np


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
            see the WPP User Guide for more information.

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
            see the WPP User Guide for more information.

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


def grid_spacing(vmin, fmax, ppw=15):
    """This function calculates the h parameter (grid_spacing)
    based on the requirement that the shortest wavelength (vmin/fmax)
    be sampled by a minimum points_per_wavelength (ppw) normally set
    to 15.
    """
    return int(vmin / (fmax * ppw))


def f_max(vmin, h, ppw=15):  # Note: ppw is regarded differently in WPP and SW4
    """Calculate teh maximum resolved frequency as a function of the
    minimum wave velocity, the grid spacing and the number of points
    per wavelength."""
    return vmin / (h * ppw)


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


def get_vmin(h, fmax, ppw=15):
    return h * fmax * ppw


def get_z(v, v0, v_grad):
    return (v - v0) / v_grad

def function():
    pass