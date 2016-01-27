"""
- wpp.py -

Python module to assist in the creation of WPP inputfile in the
pre-processing phase, run and monitor the simulation in the simulation
phase, and help read and process WPP output files in the post-
processing phase.

By: Shahar Shani-Kadmiel, August 2012, kadmiel@post.bgu.ac.il

"""

from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
import numpy as np
import fnmatch as fn
import itertools as it
import os
from shutil import copyfile
import obspy

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
############################################


############## utils #######################
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

    xmax = []; xmin = []; xrms = []; xptp = []
    for item in x:
        xmax += [item.max()]
        xmin += [item.min()]
        xrms += [rms(item)]
    xmax = max(xmax)
    xmin = min(xmin)
    xrms = rms(np.array(xrms))
    xptp = xmax-xmin
    return xmax,xmin,xrms,xptp

############################################

# This block contains pre-simulation useful functions ########################

def xy2latlon(x,y, origin=(37.0,-118.0), az=0, km2deg=111.3195):
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
    lat = origin[0] + (x*np.cos(az) - y*np.sin(az))/km2deg
    lon = (origin[1] + (x*np.sin(az) + y*np.cos(az))/
            (km2deg*np.cos(np.radians(lat))))

    return lat, lon

def latlon2xy(lat, lon, origin=(37.0,-118.0), az=0, km2deg=111.3195):
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
    x = km2deg*(
                ((lat - origin[0]) +
                 (lon - origin[1]) * np.cos(np.radians(lat))*np.tan(az)) /
                 (np.cos(az)*(1 + np.tan(az)**2)))
    y = (km2deg*((lon - origin[1]) * np.cos(np.radians(lat))) /
         np.cos(az)) - x*np.tan(az)

    return x, y

def grid_spacing(vmin, fmax, ppw=15):
    """This function calculates the h parameter (grid_spacing)
    based on the requirement that the shortest wavelength (vmin/fmax)
    be sampled by a minimum points_per_wavelength (ppw) normally set
    to 15.
    """
    return int(vmin/(fmax*ppw))

def f_max(vmin, h, ppw=15):
    """Calculate teh maximum resolved frequency as a function of the
    minimum wave velocity, the grid spacing and the number of points
    per wavelength."""
    return vmin/(h*ppw)

def f0(fmax, source_type):
    """Calculate the fundamental frequency f_0 based on fmax and the
    source type"""
    if source_type in ['Ricker', 'RickerInt', 'Gaussian', 'GaussianInt']:
        f_0 = fmax/2.5
    elif source_type in ['Brune', 'BruneSmoothed']:
        f_0 = fmax/4
    return f_0

def omega(f0, source_type):
    """Calculate omega, that value that goes on the source line in the
    WPP input file as ``freq`` based on f_0 and the source type"""
    if source_type in ['Ricker', 'RickerInt']:
        freq = f0
    elif source_type in ['Brune', 'BruneSmoothed', 'Gaussian', 'GaussianInt']:
        freq = f0*2*np.pi
    return freq

def get_vmin(h, fmax, ppw=15):
    return h*fmax*ppw

def get_z(v, v0, v_grad):
    return (v - v0)/v_grad


def trace_along_line(x1=None, x2=None, y1=None, y2=None,
            lon1=None, lon2=None, lat1=None, lat2=None,
            number_of_stations=3, name='profile', velocity=1, writeEvery=100,
            wpp_input_file=None):
    """This function places synthetic stations on a profile between
    (``x1``,``y1``) and (``x2``,``y2``) or (``lon1``,``lat1``) and (``lon2``,``lat2``).
    The number of stations is given by ``number_of_stations`` (defaults to 3).
    A ``name`` can be given as a prefix to the sac file name.
    ``velocity`` and ``writeEvery`` are WPP parameters, see WPP userguide for more info.

    The function returns a formatted string which can be copied or appended to a WPP inputfile
    if ``wpp_input_file`` is None or adds the lines to the specified file.
    """

    if x1 is not None and x2 is not None:
        x = np.linspace(x1, x2, number_of_stations)
        sac_string = 'sac x=%.3f y=%.3f depth=0 file=%s_x=%.3f_y=%.3f_ writeEvery=%d velocity=%d\n'
    elif lon1 is not None and lon2 is not None:
        x = np.linspace(lon1, lon2, number_of_stations)
        sac_string = 'sac lon=%.10f lat=%.10f depth=0 file=%s_lon=%.10f_lat=%.10f_ writeEvery=%d velocity=%d\n'

    if y1 is not None and y2 is not None:
        y = np.linspace(y1, y2, number_of_stations)
    elif lat1 is not None and lat2 is not None:
        y = np.linspace(lat1, lat2, number_of_stations)

    string = "\n\n#------------------- seismograms for traces: %s -------------------\n" %name
    for i in range(len(x)):
        string += (sac_string
                    %(x[i],y[i],name,x[i],y[i],writeEvery,velocity))

    if wpp_input_file is None:
        return string
    else:
        # copy the premade WPP input file,
        # add '+traces' to the filename
        # and open the new file in append mode
        if 'traces' not in wpp_input_file:
            filename,extention = wpp_input_file.rsplit('.',1)
            filename += '+traces.' + extention
            copyfile(wpp_input_file, filename)
        else:
            filename = wpp_input_file

        with open(filename, "a") as f:
            f.write(string)
        f.close()

        return filename

def place_station(x=None, y=None, lat=None, lon=None,
            name='st', depth=0, velocity=1, writeEvery=100,
            wpp_input_file=None):
    """This function places synthetic stations at locations (``x``,``y``) in grid coordinates
    or at (``lat``, ``lon``) in geographical coordinates.
    A sequence (list or tuple) of station names can be passed on to ``name``.
    ``velocity`` and ``writeEvery`` are WPP parameters, see WPP userguide for more info.

    The function returns a formatted string which can be copied or appended to a WPP inputfile
    if ``wpp_input_file`` is None or adds the lines to the specified file.
    """

    if x is not None and y is not None:
        sac_string = 'sac x=%.3f y=%.3f depth=0 file=%s writeEvery=%d velocity=%d\n'
    elif lon is not None and lat is not None:
        x,y = lon,lat
        sac_string = 'sac lon=%.5f lat=%.5f depth=0 file=%s writeEvery=%d velocity=%d\n'

    string = "\n\n#------------------- seismogram stations: -------------------\n"
    try:
        for i in range(len(x)):
            string += (sac_string
                        %(x[i],y[i],name[i],writeEvery,velocity))
    except TypeError:
        string += (sac_string
                        %(x,y,name,writeEvery,velocity))

    if wpp_input_file is None:
        return string
    else:
        # copy the premade WPP input file,
        # add '+stations' to the filename
        # and open the new file in append mode
        if 'stations' not in wpp_input_file:
            filename,extention = wpp_input_file.rsplit('.', 1)
            filename += '+stations.' + extention
            copyfile(wpp_input_file, filename)
        else:
            filename = wpp_input_file

        with open(filename, "a") as f:
            f.write(string)
        f.close()

        return filename

def revise_inputfile(replacements, original, revised=None):
    """This function searches the inputhfile for the strings
    in replacements and performs the appropriate substitutions.
    replacements should be a dictionary of the form:

    {'original string 1':'substitusion string 1',
     'original string 2':'substitusion string 2',
     ...
     }

    """
    infile = open(original)

    if revised is None:
        revised = original + '.temp'

    outfile = open(revised, 'w')

    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target, 1)
        outfile.write(line)
    infile.close()
    outfile.close()
    os.rename(revised, original)



##############################################################################

# This block contains co-simulation useful functions #########################

def visual(watch_for, path_to_watch="./", wait=10):
    """A function to monitor the output directory ("path_to_watch") every
    "wait" seconds (default is 10) and refresh the display with the latest
    image file or files in "watch_for" (single or list of files)."""

    before = [f for f in os.listdir (path_to_watch)]
    while 1:
        time.sleep(wait)
        after = [f for f in os.listdir (path_to_watch)]
        added = [f for f in after if not f in before]
        if watch_for in added:
            print watch_for, "was added and is being plotted..."
            #here is where the plotting happens...

        before = after

##############################################################################


# This block contains post-simulation useful functions #######################

## Image related functions and classes

def readhdr(filename, verbose=True, n_o_p=None, p_i=None):
    """ This function reads a WPP image file header:

            Usage: precision, number_of_patches, patch_info, position = readhdr(filename)

            where "filename" is the name of the image file."""

    # open the image file for binary reading
    f = open(filename, 'rb')

    # read the first two ints from file
    # (precision, # of patches)
    precision, number_of_patches = np.fromfile(f, dtype='int32', count=2)

    # a patch header is made up of 1 double and 4 ints
    # (spacing, i start, i end, j start, j end)
    patch_info_dtype = np.dtype([
        ('h','float64'),
        ('ib','int32'),('ie','int32'),
        ('jb','int32'),('je','int32')])

    # read information from # of patches as defined above
    patch_info = np.fromfile(f, dtype=patch_info_dtype, count=number_of_patches)

    if precision == 4:
        p = "single"
    elif precision == 8:
        p = "double"
    elif precision == 0 and n_o_p is None:
        print '''Error: File header is empty.
Try passing the header information via the defaults keywords:
    n_o_p : int
    p_i : [float, int,int,int,int]'''
    elif precision == 0 and n_o_p is not None:
        precision = 4
        p = '0'
        print 'Overriding header...'
        number_of_patches = n_o_p
        patch_info = p_i

    if verbose:
        print ("readhdr:    Processing a " + p +
               " precision image file with " + str(number_of_patches) + " patches.")

    return precision, number_of_patches, patch_info, f.tell()

def readonepatch(filename, precision=-1, nx=-1, ny=-1, position=-1, verbose=True):
    """ This function reads a patch from a WPP image file:

        Usage: (data, position) = readonepatch(filename, precision, nx, ny, position)

        where "filename" is the name of the image file."""

    # open the image file for binary reading
    f = open(filename, 'rb')

    # seek to the right place in the file
    f.seek(position)

    # read patch data
    if precision == 4:
        data = np.fromfile(f, dtype='float32', count=((nx)*(ny)))
    else:
        data = np.fromfile(f, dtype='float64', count=((nx)*(ny)))

    if verbose:
        print ("readonepatch:   Patch max is " + str(data.max()) +
                ", patch min is " + str(data.min()))

    return data.reshape(ny, nx), f.tell()

def read(filename, grid_filename=None, coordinate_files=None, dt=None, verbose=False, n_o_p=None, p_i=None):
    """Read WPP image file into a WPP CrossSection or Map object and returns
    an instance on that object.

    ***NOTE***

        if you are only interested in the data, consider using
        `wpp.raw_read` which is much faster instead.

    Parameters
    -----------
    filename : string, name of the WPP image file

    grid_filename : bool** or string, optional, default: None
        when topography is used the Zgrid file must be
        supplied in order to plot the curvilinear grid correctly.
        ** See note at the bottom about the bool option **

    coordinate_files : bool**, tuple or list of strings, optional, default: None
        when the computational domain is rotated 'az' degrees (see the WPP
        User Guide) and/or the data should be plotted in geographical coordinates
        rather than cartesian, (.lon, .lat) files must be supplied.
        ** See note at the bottom about the bool option **

    dt : float, otional, default: None
        if supplied, the time-stamp of the timestep will be plotted to the
        bottom right corner of the image

    verbose : bool, optional, default: False
        set to True if you want to see what's going on.
        Warning! may generate a lot of output.

    ***ANOTHER NOTE***

        if image file is a map and coordinate_files=True:
            Coordinate files will be automatically looked up in the same
            path as the image file. The code looks for files with a .lon
            and a .lat extention.

        if image is a cross-section and grid_filename=True:
            Grid file will be automatically looked up in the same
            path as the image file. The code looks for a file with a .grid
            extention and makes sure that the slicing coordinate matches
            that of the image file. For example: for image file named
            yzplane.cycle=1750.x=123456.velmag' the file '*Z*x=123456.grid'.
    """

    filename = os.path.abspath(filename)
    base, name = os.path.split(filename)
    file_dict, label_dict = get_type(filename)

    if file_dict['data_type'] == "cross section":
        #look for grid_file
        if grid_filename is True:
            Zgrid = None
            for f in os.listdir(base):
                if fn.fnmatch(f,
                              '*Z*%s=%s.grid' %(file_dict['plane'],
                                                file_dict['plane_value'])):
                    grid_filename = os.path.join(base, f)
                    break
        wppfile = CrossSection(filename, file_dict, label_dict,
                                grid_filename, dt, verbose, n_o_p, p_i)
    elif file_dict['data_type'] == 'map':
        #look for coordinate_files
        if coordinate_files is True:
            lonfile, latfile = None, None
            for f in os.listdir(base):
                if fn.fnmatch(f, '*.lon') and lonfile is None:
                    lonfile = os.path.join(base, f)
                if fn.fnmatch(f, '*.lat') and latfile is None:
                    latfile = os.path.join(base, f)
                if lonfile is not None and latfile is not None:
                    coordinate_files = (lonfile, latfile)
                    break
        wppfile = Map(filename, file_dict, label_dict,
                      coordinate_files, dt, verbose, n_o_p, p_i)

    return wppfile

def raw_read(filename, verbose=False, n_o_p=None, p_i=None):
    """Reads WPP image file into a WPP ImageFile object

    Parameters
    -----------
    filename : string, name of the WPP image file

    verbose : bool, optional, default: False
        set to True if you want to see what's going on.
        Warning! may generate a lot of output.
    """

    wppfile = ImageFile(filename, verbose, n_o_p, p_i)
    return wppfile


def load_grid(filename):
    """Reads WPP gridfile into a WPP Grid object

    Parameters
    -----------
    filename : string, name of the WPP gridfile

    verbose : bool, optional, default: False
        set to True if you want to see what's going on.
        Warning! may generate a lot of output.
    """

    grid = Grid(filename)
    return grid

def parse_filename(filename):
    """ This function parses the filename in order to figure out its type.

    Parameters
    -----------
    filename : string, name of the WPP file

    Returns
    --------
    name, cycle, plane, plane_value, extention
    """

    name, cycle, plane, extention = filename.rsplit('.', 3)
    name = name.rsplit('/', 1)[-1]
    cycle = int(cycle.split('=')[-1])
    plane, plane_value = plane.split('=')

    return name, cycle, plane, plane_value, extention

def get_type(filename):
    """This finction sets the name of the dictionary to use based on the
        contents of the filename and the extention.

    Parameters
    -----------
    filename : string, name of the WPP file

    Returns
    --------
    file_dict, label_dict
    """

    name, cycle, plane, plane_value, extention = parse_filename(filename)

    file_keys = ('name', 'cycle', 'plane', 'plane_value',
            'extention', 'data_in_file', 'data_type')

    # set default label_vals incase one is not set by the options below
    label_keys = ('title', 'xlabel', 'ylabel', 'colorbarlabel')
    label_vals = (None, None, None, None)

    # check if cross-section
    if plane == 'x' or plane == 'y':
        data_type = 'cross section'
        xlabel = 'Distance along ' + plane + '=' + plane_value + ', km'
        if extention == 'grid':
            data_in_file = 'grid'
        elif extention == 'p':
            data_in_file = 'Pressure-waves velocity model'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', 'P-waves velocity, km/s')
        elif extention == 's':
            data_in_file = 'Shear-waves velocity model'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', 'S-waves velocity, km/s')
        elif extention == 'rho':
            data_in_file = 'Density model'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', 'Density, gr/cm^3')
        elif extention == 'qp':
            data_in_file = 'Qp model'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', 'Qp')
        elif extention == 'qs':
            data_in_file = 'Qs model'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', 'Qs')
        elif extention == 'ux' or extention == 'uy' or extention == 'uz':
            data_in_file = extention
            label_vals = (data_in_file, xlabel, 'Depth, km', data_in_file + ', m')
        elif extention == 'div': # no S-waves
            data_in_file = extention + ' u'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file)
        elif extention == 'curl': # no P-waves
            data_in_file = extention + ' u'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file)
        elif extention == 'veldiv':
            data_in_file = 'd/dt div u'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', 1/s')
        elif extention == 'velcurl':
            data_in_file = 'd/dt curl u'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', 1/s')
        elif extention == 'velmag':
            data_in_file = 'velocity magnitude'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', m/s')
        elif extention == 'hvel':
            data_in_file = 'horizontal velocity magnitude'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', m/s')
        elif extention == 'hvelmax':
            data_in_file = 'horizontal velocity maximum'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', m/s')
        elif extention == 'vvelmax':
            data_in_file = 'vertical velocity maximum'
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', m/s')
        elif extention == 'fx' or extention == 'fy' or extention == 'fz':
            data_in_file = extention
            label_vals = (data_in_file, 'Distance, km', 'Depth, km', data_in_file + ', N')

    # check if map
    # x and y lables will be overwritten if coordinates files are supplied
    elif plane == 'z':
        data_type = 'map'
        if extention == 'grid':
            data_in_file = 'topography'
            label_vals = (data_in_file, 'y, km', 'x, km', 'm.a.s.l')
        elif extention == 'lat':
            data_in_file = 'latitude'
        elif extention == 'lon':
            data_in_file = 'longitude'
        elif extention == 'p':
            data_in_file = 'Pressure-waves velocity model'
            label_vals = (data_in_file, 'y, km', 'x, km', 'P-waves velocity, km/s')
        elif extention == 's':
            data_in_file = 'Shear-waves velocity model'
            label_vals = (data_in_file, 'y, km', 'x, km', 'S-waves velocity, km/s')
        elif extention == 'rho':
            data_in_file = 'Density model'
            label_vals = (data_in_file, 'y, km', 'x, km', 'Density, gr/cm^3')
        elif extention == 'qp':
            data_in_file = 'Qp model'
            label_vals = (data_in_file, 'y, km', 'x, km', 'Qp')
        elif extention == 'qs':
            data_in_file = 'Qs model'
            label_vals = (data_in_file, 'y, km', 'x, km', 'Qs')
        elif extention == 'ux' or extention == 'uy' or extention == 'uz':
            data_in_file = extention
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', m')
        elif extention == 'div': # no S-waves
            data_in_file = extention + ' u'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file)
        elif extention == 'curl': # no P-waves
            data_in_file = extention + ' u'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file)
        elif extention == 'veldiv':
            data_in_file = 'd/dt div u'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', 1/s')
        elif extention == 'velcurl':
            data_in_file = 'd/dt curl u'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', 1/s')
        elif extention == 'velmag':
            data_in_file = 'velocity magnitude'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', m/s')
        elif extention == 'hvel':
            data_in_file = 'horizontal velocity magnitude'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', m/s')
        elif extention == 'hvelmax':
            data_in_file = 'horizontal velocity maximum'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', m/s')
        elif extention == 'vvelmax':
            data_in_file = 'vertical velocity maximum'
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', m/s')
        elif extention == 'fx' or extention == 'fy' or extention == 'fz':
            data_in_file = extention
            label_vals = (data_in_file, 'y, km', 'x, km', data_in_file + ', N')

    file_vals = (name, cycle, plane, plane_value,
            extention, data_in_file, data_type)

    file_dict = dict(zip(file_keys, file_vals))

    label_dict = dict(zip(label_keys, label_vals))

    return file_dict, label_dict

def get_corners(extent):
    """Get the verticies (5) of the square polygon defined by extent.
    extent is in the form (xmin, xmax, ymin, ymax)

        Usage: cornersx, cornersy = get_corners(extent)
    """

    corners = list(it.product(extent[:2],extent[2:]))
    corners[2], corners[3] = corners[3], corners[2]
    corners.append(corners[0])
    cornersx = [corner[0] for corner in corners]
    cornersy = [corner[1] for corner in corners]

    return cornersx, cornersy

class ImageFile(object):
    """
    The ImageFile class holds a single- or multi-patch WPP file without data_type considerations.
    It is created by the raw_read function above and needs a filename and a file_dict.

    """
    def __init__(self, filename, verbose, n_o_p, p_i):

        self.filename = filename

        # read the image header
        (precision,
            self.number_of_patches,
            self.patch_info,
            position) = readhdr(self.filename, verbose, n_o_p, p_i)

        # read and store patch/s data
        self.patches = []
        for i, patch in enumerate(self.patch_info):
            h = patch[0]
            nx = (patch[2]-patch[1])+1
            ny = (patch[4]-patch[3])+1

            # read each patch and append to the list of arrays at index 'i'
            if verbose:
                print ("read:   Reading patch " + str(i) + ":" )
            data, position = readonepatch(self.filename,
                                                precision,
                                                nx, ny,
                                                position,
                                                verbose)
            self.patches.append(data)

class CrossSection(ImageFile):
    """
    The CrossSection class holds a single- or multi-patch WPP cross section file.
    It is created by the read function above and needs a filename and a file_dict.
    The grid_filename is optional but is necessary in order to correctly plot curvilinear
    grids when the topography command is used with WPP. If no grid_filename is given,
    one is created from the information in the patch_info array.

    """
    def __init__(self, filename, file_dict, label_dict, grid_filename, dt, verbose, n_o_p, p_i):
        ImageFile.__init__(self, filename, verbose, n_o_p, p_i)
        self.file_dict = file_dict
        self.label_dict = label_dict
        self.grid_filename = grid_filename
        self.dt = dt

        if self.grid_filename is None:
            self.grid = Grid(patch_info=self.patch_info, verbose=verbose)
        else:
            if verbose:
                print self.grid_filename
            self.grid = load_grid(self.grid_filename)

        self.extent = self.grid.extent

        # modify patch data
        for patch in self.patches:
            if (self.file_dict['data_in_file'] in
                ['Pressure-waves velocity model',
                'Shear-waves velocity model']):
                patch *= 1e-3

        # make a 1Darray of the data, calculate some stuff
        self.max,self.min,self.rms,self.ptp = calc_stuff(self.patches)

    def plot(self, title=None, size=(4,4), ax=None,
                contours=None, cmap=plt.cm.jet, clipfactor=3, shading='flat', alpha=1,
                colorbar=True, annotate=True,
                save=False, **kwargs):
        """Plotting method for WPP CrossSection class object.
        shading can be either 'flat' (faster, default) or 'gouraud'
        contours defaults to None but can be given as an int value or a tuple of clevels"""

        return_array = [ax]
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            return_array.insert(0, fig)

        if clipfactor is 'max':
            clip = self.max
        elif type(clipfactor) is float:
            clip = clipfactor
        else:
            clip = clipfactor*self.rms

        vmin = -clip
        vmax = clip
        if ('mag' in self.file_dict['extention'] or
            'velmax' in self.file_dict['extention'] or
            'p' in self.file_dict['extention'] or
            's' in self.file_dict['extention'] or
            'rho' in self.file_dict['extention']):
            vmin = 0

        im = []

        lw = 1

        for i in range(self.number_of_patches-1,-1,-1):
            x = self.grid.Hpatches[i]*1e-3
            y = self.grid.Zpatches[i]*1e-3
            data = self.patches[i]

            print("Plotting patch %d of %d patches" %(i, self.number_of_patches))
            im.append(ax.pcolormesh(x, y, data,
                      cmap=cmap, shading=shading,
                      vmin=vmin, vmax=vmax, alpha=alpha))

            if contours:
                cs = ax.contour(x, y, data, contours, colors='k',
                        vmin=self.min, vmax=self.max)
                ax.clabel(cs, fmt='%.2f', fontsize=10, rightside_up=True)

            if self.number_of_patches > 1:
                ax.plot(x[0],y[0],'k',linewidth=lw)
                lw = 0.5

        return_array.append(im)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im[0], cax=cax,
                              extend=('max' if clipfactor is not 'max' else 'neither'))
            cb.set_label(self.label_dict['colorbarlabel'])
            cb.formatter.set_scientific(True)
            cb.formatter.set_powerlimits((-1,4))
            cb.update_ticks()
            return_array.append(cb)

        ax.set_xlim(self.extent[0],self.extent[1])
        ax.set_ylim(self.extent[2],self.extent[3])
        ax.set_aspect('equal')

        ax.set_xlabel(self.label_dict['xlabel'])
        ax.set_ylabel(self.label_dict['ylabel'])

        if title is None:
            ax.set_title(self.label_dict['title'], fontsize=14, y=1.03)
        elif title is 'off':
            pass
        else:
            ax.set_title(title, fontsize=14, y=1.03)

        # put a time stamp in the bottom right corner of the image
        if self.dt:
            time_stamp = ('t = %.2f s' % (self.file_dict['cycle']*self.dt))
            stamp = ax.text(0.96, 0.03, time_stamp,
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=14)
            stamp.set_path_effects(
                [pe.Stroke(linewidth=3, foreground="w"), pe.Normal()])

        # add textual information about the plot
        if annotate:
            info_string = (self.file_dict['data_in_file'] + ', Plane: ' + self.file_dict['plane'] + '=' + self.file_dict['plane_value'])
            info_string += (', Time step: ' + str(self.file_dict['cycle']))
            if self.dt is not None:
                info_string += (', Time: %s' % time_stamp)

            ax.text(0.5, -0.2, info_string, horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)

        # save the figure
        if save:
            fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)

        return return_array

class Map(ImageFile):
    """
    The Map class holds a WPP surface file conforming to some z level.
    It is created by the read function above and needs a filename and a file_dict.
    The coordinate_files is a tuple of 2 filenames, longitude and latitude,
    holding the geographical coordinates of the data in the map and are optional.
    If no coordinate files are given the data are considered on the x, y cartesian
    grid in the patch_info array.

    """

    def __init__(self, filename, file_dict, label_dict, coordinate_files, dt, verbose, n_o_p, p_i):
        ImageFile.__init__(self, filename, verbose, n_o_p, p_i)
        self.file_dict = file_dict
        self.label_dict = label_dict
        self.coordinate_files = coordinate_files
        self.dt = dt

        if self.coordinate_files:
            self.label_dict['xlabel'] = 'lon.'
            self.label_dict['ylabel'] = 'lat.'
            if verbose:
                print "Geographical coordinates supplied..."
                print("longitudes file: %s" %self.coordinate_files[0])
                print("latitudes file: %s" %self.coordinate_files[1])
            self.lon = raw_read(self.coordinate_files[0])
            self.lat = raw_read(self.coordinate_files[1])

            self.extent = (self.lon.patches[0].min(), self.lon.patches[0].max(),
                            self.lat.patches[0].min(), self.lat.patches[0].max())
        else:
            if verbose:
                print("No geographical coordinates supplied, using cartesian coordinates in header...")
            patch = self.patch_info[0]
            h = patch[0]
            nx = (patch[2]-patch[1])+1
            ny = (patch[4]-patch[3])+1
            self.extent = (0, h*ny*1e-3, 0, h*nx*1e-3)

        # read and store patch/s data
        for patch in self.patches:
            if self.file_dict['data_in_file'] == 'topography':
                data *= -1
            if (self.file_dict['data_in_file'] in
                ['Pressure-waves velocity model',
                'Shear-waves velocity model']):
                patch *= 1e-3

        # make a 1Darray of the data, calculate some stuff
        self.max,self.min,self.rms,self.ptp = calc_stuff(self.patches)

    def plot(self, title=None, size=(4,4), ax=None,
                contours=None, cmap=plt.cm.jet, clipfactor=3, shading='flat', alpha=1,
                colorbar=True, annotate=True, mask=None,
                save=False, **kwargs):
        """Plotting method for WPP Map class object.
        shading can be either 'flat' (faster, default) or 'gouraud'
        contours defaults to None but can be given as an int value
        or a tuple of clevels"""

        return_array = [ax]
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            return_array.insert(0, fig)

        if clipfactor is 'max':
            clip = self.max
        elif type(clipfactor) is float:
            clip = clipfactor
        else:
            clip = clipfactor*self.rms

        vmin = -clip
        vmax = clip
        if ('mag' in self.file_dict['extention'] or
            'velmax' in self.file_dict['extention'] or
            'p' in self.file_dict['extention'] or
            's' in self.file_dict['extention'] or
            'rho' in self.file_dict['extention']):
            vmin = 0

        data = self.patches[0]
        if mask:
            if type(mask) is float:
                value = mask
            elif type(mask) is int:
                value = mask*self.rms

            data = np.ma.masked_less_equal(data, value)

        if self.file_dict['data_in_file'] == 'topography':
            vmin = data.min()

        if self.coordinate_files:
            x = self.lon.patches[0]
            y = self.lat.patches[0]
            im = ax.pcolormesh(x, y, data, cmap=cmap, shading=shading,
                        vmin=vmin, vmax=vmax, alpha=alpha)
            if contours:
                cs = ax.contour(x, y, data, contours, colors='k',
                    vmin=self.min, vmax=self.max)
                ax.clabel(cs, fmt='%.2f', fontsize=10, rightside_up=True)
        else:
            if shading is 'flat':
                interpolation = 'nearest'
            else:
                interpolation = 'bicubic'
            im = ax.imshow(data.T, cmap=cmap, extent=self.extent,
                interpolation=interpolation, vmin=vmin, vmax=vmax, alpha=alpha,
                origin='lower', rasterized=True)
            if contours:
                print('contours=%s' %contours)
                cs = ax.contour(data.T, contours, colors='k', extent=self.extent,
                    origin='lower', vmin=self.min, vmax=self.max)
                ax.clabel(cs, fmt='%.2f', fontsize=10, rightside_up=True)

        return_array.append(im)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im, cax=cax,
                              extend=('max' if clipfactor is not 'max' else 'neither'))
            cb.set_label(self.label_dict['colorbarlabel'])
            cb.formatter.set_scientific(True)
            cb.formatter.set_powerlimits((-1,4))
            cb.update_ticks()
            return_array.append(cb)

        #ax.axis(self.extent)
        ax.set_aspect('equal')

        ax.set_xlabel(self.label_dict['xlabel'])
        ax.set_ylabel(self.label_dict['ylabel'])

        if title is None:
            ax.set_title(self.label_dict['title'], fontsize=14, y=1.03)
        elif title is 'off':
            pass
        else:
            ax.set_title(title, fontsize=14, y=1.03)

        # put a time stamp in the bottom right corner of the image
        if self.dt:
            time_stamp = ('t = %.2f s' % (self.file_dict['cycle']*self.dt))
            stamp = ax.text(0.96, 0.03, time_stamp,
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=14)
            stamp.set_path_effects(
                [pe.Stroke(linewidth=3, foreground="w"), pe.Normal()])

        # add textual information about the plot
        if annotate:
            info_string = (self.file_dict['data_in_file'] + ', Plane: ' + self.file_dict['plane'] + '=' + self.file_dict['plane_value'])
            info_string += (', Time step: ' + str(self.file_dict['cycle']))
            if self.dt is not None:
                info_string += (', Time: %s' % time_stamp)

            ax.text(0.5, -0.2, info_string, horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)

        # save figure
        if save:
            fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)

        return return_array

class Grid(ImageFile):
    """
    WPP grid file representing a multi-patch cross-section grid file
    needed for plotting multi-patch images. If topography is present
    the file consists of 1 curvilinear grid and at least 1 other
    cartesian grid depending on the number of mesh refinements used
    in the simulation.
    If no filename is given, then a cartesian grid is formed instead.
    Needs only the name of the WPP Z grid file.
    """
    def __init__(self, filename=None, patch_info=None, verbose=False):
        if filename:
            ImageFile.__init__(self, filename, verbose)
            self.Zpatches = self.patches
            self.Hpatches = []
            for patch in self.patch_info:
                h = patch[0]
                nx = (patch[2]-patch[1])+1
                ny = (patch[4]-patch[3])+1

                xx = (np.ones((nx,ny)).T*(np.linspace(0,(nx-1)*h,nx)).T)
                self.Hpatches += [xx]
        else:
            # make and store patch/s data
            if verbose:
                print "Making a grid..."
            self.Zpatches = []
            self.Hpatches = []

            total_depth = 0.
            for patch in patch_info:
                h = patch[0]
                ny = (patch[4]-patch[3])
                total_depth += ny*h

            patch_bottom = total_depth
            for patch in patch_info:
                h = patch[0]
                nx = (patch[2]-patch[1])+1
                ny = (patch[4]-patch[3])+1

                patch_top = patch_bottom - (ny-1)*h
                x = np.linspace(0,(nx-1)*h,nx)
                y = np.linspace(patch_top,patch_bottom,ny)
                xx,yy = np.meshgrid(x,y)
                self.Zpatches += [yy]
                self.Hpatches += [xx]

                patch_bottom = patch_top

        self.Z_max = max([patch.max() for patch in self.Zpatches])
        self.Z_min = min([patch.min() for patch in self.Zpatches])

        self.H_max = max([patch.max() for patch in self.Hpatches])
        self.H_min = min([patch.min() for patch in self.Hpatches])

        self.extent = np.array([self.H_min, self.H_max, self.Z_max, self.Z_min]) * 1e-3

    def plot(self, size=(4,4), save='no', **kwargs):
        """Plot method for the Grid class"""

        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0,0,1,1])

        color = 'r'
        for i in range(self.number_of_patches-1,-1,-1):
            hgrid = self.Hpatches[i] * 1e-3
            zgrid = self.Zpatches[i] * 1e-3

            #vertical lines
            ax.plot(hgrid, zgrid, color, linewidth=0.5)

            #horizontal lines
            ax.plot(hgrid.T, zgrid.T, color, linewidth=0.5)

            color = 'b'

        ax.set_xlim(self.extent[0],self.extent[1])
        ax.set_ylim(self.extent[2],self.extent[3])
        ax.set_aspect('equal')

        ax.set_xlabel('Distance, km')
        ax.set_ylabel('Depth, km')
        ax.set_title('Cross section grid view', fontsize=14, y=1.03)

        return fig, ax


def xvelmax2PGV(hvelmax, vvelmax, dt=None):
    """
    This function computes PGV such that:

        PGV = SQRT(HPGV^2 + VPGV^2)

    returns
    --------
    a WPP ImageFile object with the new data

    Parameters
    -----------
    hvelmax and vvelmax : are either WPP ImageFile objects or
        filenames to the WPP .hvelmax and .vvelmax files
    """

    try:
        Himage = read(hvelmax, None, None, dt)
        Vimage = read(vvelmax, None, None, dt)
    except AttributeError:
        Himage = hvelmax
        Vimage = vvelmax

    newimage = deepcopy(Himage)
    (name,
     cycle,
     plane,
     plane_value,
     extention) = parse_filename(Himage.filename)
    newimage.filename = None
    try:
        newimage.file_dict['data_in_file'] = 'PGV'
        newimage.file_dict['extention'] = 'xvelmax'
        newimage.label_dict['colorbarlabel'] = 'PGV, m/s'
        newimage.label_dict['title'] = 'Peak ground velocity'
    except AttributeError:
        newimage.file_dict = {'cycle'        : cycle,
                              'data_in_file' : 'PGV',
                              'extention'    : 'xvelmax'}
        newimage.label_dict = {'colorbarlabel' : 'PGV, m/s',
                               'title'         : 'Peak ground velocity'}

    if dt is not None:
        newimage.dt = dt

    for i in range(Himage.number_of_patches):
        Hdata = Himage.patches[i]
        Vdata = Vimage.patches[i]
        newimage.patches[i] = np.sqrt(Hdata**2 + Vdata**2)

    (newimage.max,
     newimage.min,
     newimage.rms,
     newimage.ptp) = calc_stuff(newimage.patches)

    return deepcopy(newimage)



## Input file and monitor file related functions and classes

def parse_inputfile(filename, verbose=False):
    """
    This fuction creates the Input_file class which parses the
    WPP input file.

        Usage: inputfile = parse_inputfile(filename)
    """

    return Input_file(filename, verbose)

class Input_file(object):
    """
    The Input_file class holds relevant information found in the
    WPP input file.
    """

    def __init__(self, filename, verbose):

        self.filename = filename

        with open(filename, 'r') as f:
            dict_name = 'None_None'
            i = 1
            for j,line in enumerate(f):
                line = line.strip().rstrip()

                if verbose:
                    print j, ':', line

                if line.startswith('#') or len(line) < 2:
                    continue
                else:
                    words = line.split()
                    if words[0] == dict_name.split('_')[0]:
                        dict_name = words.pop(0) + '_' + str(i)
                        i += 1
                    else:
                        i = 1
                        dict_name = words.pop(0)

                    vars(self)[dict_name] = {}

                    for word in words:
                        key, val = word.split('=', 1)
                        vars(self)[dict_name][key] = val

def parse_monitor(filename):
    """
    This function creates the Simulation_metadata class which parses
    the monitor text file written during the simulation runtime.

        Usage: metatata = parse_monitor(filename)
    """

    return Simulation_metadata(filename)

class Simulation_metadata(object):
    """
    The Simulation_metadata class holds information found in the monitor
    text file written during simulation runtime. Information includes timing
    parameters of different phases of the simulation, grid points count etc.
    """

    def __init__(self, filename):

        self.filename = filename

        with open(filename, 'r') as f:
            minPPW = []
            for line in f:
                if "* Setting nx to" in line:
                    self.nx = int(line.split()[4])
                if "* Setting ny to" in line:
                    self.ny = int(line.split()[4])
                if "* Setting nz to" in line:
                    self.nz = int(line.split()[4])
                if "input phase" in line:
                    self.input_phase = float(line.split()[-2])
                if "start up phase" in line:
                    self.startup_phase = float(line.split()[-2])
                if "Start Time =" in line:
                    self.start_time = float(line.split()[3])
                    self.goal_time = float(line.split()[-1])
                if "minVs/h=" in line:
                    minPPW.append(float(line.split()[2].split('=')[-1]))
                if "max freq=" in line:
                    self.maxfreq = float(line.split()[1].split('=')[-1])
                if "Total seismic moment" in line:
                    self.M0 = float(line.split()[4])
                if "Moment magnitude" in line:
                    self.Mw = float(line.split()[3])
                if "Number of sources" in line:
                    self.sources = float(line.split()[3])
                if "Running program WPP" in line:
                    self.number_of_cpus = int(line.split()[4])
                if "Number of time steps" in line:
                    self.npts = int(line.split()[5])
                    self.dt = float(line.split()[-1])
                if "solver phase" in line:
                    self.solver_phase = 0.0
                    line_arr = line.split()
                    if "hour" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('hour')-1]) * 3600
                    elif "hours" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('hours')-1]) * 3600
                    if "minute" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('minute')-1]) * 60
                    elif "minutes" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('minutes')-1]) * 60
                    if "second" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('second')-1])
                    elif "seconds" in line_arr:
                        self.solver_phase += float(line_arr[line_arr.index('seconds')-1])
                if "Total number of grid points" in line:
                    self.number_of_grid_points = float(line.split()[-1])

            self.minPPW = int(min(minPPW)/self.maxfreq)

## Signal processing related functions and classes

def sortkey(f):
    """ This function is a service for read_traces which
    helps sort the filenames correctly so that the traces
    data is read in the right order along the profile.
    """
    xi, yi = f.split('_')[1:3]
    sigma = float(xi.split('=')[1]) + float(yi.split('=')[1])

    return sigma

def read_traces(results_dir, name, components=['xv','yv','zv']):
    """

    """

    traces = Traces()

    profile = []
    if type(name) is str:
        files = os.listdir(results_dir)
        for filename in files:
            if name in filename:
                profile.append(filename)
    elif type(name) is list:
        profile = name

    profile.sort(key=sortkey)

    for comp in components:
        vars()['traces_' + comp] = []
    traces_xi = []; traces_yi = [];
    for i,comp in enumerate(components):
        for f in profile:
            if f.endswith(comp):
                filename = '/'.join([results_dir,f])
                tr = obspy.read(filename, format='sac')[0]
                data = tr.data
                vars()['traces_' + comp].append(data)
                if i == 0:
                    # get the coordinates of each station only once
                    xi, yi = f.split('_')[1:3]
                    xcoor, xi = xi.split('=')
                    ycoor, yi = yi.split('=')
                    traces_xi.append(float(xi))
                    traces_yi.append(float(yi))

    delta = tr.stats.delta
    npts = tr.stats.npts
    time = tr.times()

    for comp in components:
        vars(traces)[comp] = np.asarray(vars()['traces_' + comp]).T
    traces_xi = np.asarray(traces_xi)
    traces_yi = np.asarray(traces_yi)

    distance = np.sqrt(traces_xi.ptp()**2 + traces_yi.ptp()**2)
    distance = np.linspace(0, distance*1e-3, traces_xi.size)

    di, ti = np.meshgrid(distance, time)

    # Assign to the Traces class
    traces.name         = name
    traces.components   = components
    traces.delta        = delta
    traces.npts         = npts
    #traces.xi          = traces_xi
    #traces.yi          = traces_yi
    traces.time         = time
    traces.distance     = distance
    traces.di           = di
    traces.ti           = ti

    print traces
    return traces


class Traces(object):
    """
    """

    def __init__(self):
        pass

    def __str__(self):
        """
        """

        out = 'Trace profile %s in %d %s component' %(self.name, len(self.components), self.components)
        if len(self.components) > 1:
            out += 's:'
        else:
            out += ':'

        return out

    def plot_traces(self, component, ax=None, clipfactor=3,
             cmap=plt.cm.gray_r, size=(4,4), colorbar=True, shading='gouraud',
             wiggls=True, decimate_by=None, color='r',
             xmajor_every=None, xminor_every=None,
             ymajor_every=None, yminor_every=None,
             title=None, save=False, **kwargs):
        """
        """

        data = vars(self)[component]

        if clipfactor is 'max':
            clip = abs(data).max()
        elif type(clipfactor) is float:
            clip = clipfactor
        else:
            clip = clipfactor*data.std()

        return_array = [ax]
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            return_array.insert(0, fig)

        im = ax.pcolormesh(self.di, self.ti, data,
                            cmap=cmap, vmin=-clip, vmax=clip, shading=shading,
                            **kwargs)

        if wiggls:
            if decimate_by is None:
                decimate_by = int(0.1*self.distance.size)
            selected = data.T[2*decimate_by:-2*decimate_by:decimate_by]
            for i,line in enumerate(selected):
                ax.plot(self.distance[2*decimate_by:-2*decimate_by:decimate_by][i]+line*wiggls,
                 self.time, color, zorder=3)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im, cax,
                            extend=('both' if clipfactor is not 'max' else 'neither'))
            cb.set_label(component[0] + ' comp. ground velocity, m/s')
            cb.formatter.set_scientific(True)
            cb.formatter.set_powerlimits((-1,4))
            cb.update_ticks()

        if xmajor_every is None:
            xmajor_every = round(self.distance.max()/4.)
        if xminor_every is None:
            xminor_every = xmajor_every/2.
        if ymajor_every is None:
            ymajor_every = round(self.time.max()/4.)
        if yminor_every is None:
            yminor_every = ymajor_every/2.


        xmajorLocator = MultipleLocator(xmajor_every)
        xminorLocator = MultipleLocator(xminor_every)

        ax.set_xlim(self.distance[0], self.distance[-1])
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.set_xlabel('Distance, km')

        ymajorLocator = MultipleLocator(ymajor_every)
        yminorLocator = MultipleLocator(yminor_every)

        ax.set_ylim(self.time[-1], 0)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.set_ylabel('Time, s')

        if title:
            ax.set_title(title, y=1.03)

        if save:
            fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)
        else:
            if ax is None: fig.show()

        return return_array

    def plot_spectral_image(self, component, ax=None, maxfreq=None, freq_scale='log',
                            cmap=plt.cm.jet, clipfactor=3,
                            size=(4,4), colorbar=True, shading='gouraud',
                            xmajor_every=None, xminor_every=None,
                            title=None, save=False, **kwargs):
        """
        """

        data = vars(self)[component]
        self.amplitude = np.abs(np.fft.rfft(data, axis=0))/(0.5*self.npts)
        self.frequency = np.fft.rfftfreq(self.npts, self.delta)

        self.dfi, self.fi = np.meshgrid(self.distance, self.frequency)

        if clipfactor is 'max':
            clip = abs(self.amplitude).max()
        elif type(clipfactor) is float:
            clip = clipfactor
        else:
            clip = clipfactor*self.amplitude.std()


        return_array = [ax]
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            return_array.insert(0, fig)

        im = ax.pcolormesh(self.dfi, self.fi, self.amplitude, shading=shading, cmap=cmap,
                           vmin=0, vmax=clip, **kwargs)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im, cax,
                        extend=('max' if clipfactor is not 'max' else 'neither'))
            cb.set_label(component[0] + ' comp. spectral velocity, m/s')
            cb.formatter.set_scientific(True)
            cb.formatter.set_powerlimits((-1,4))
            cb.update_ticks()

        if maxfreq is None:
            maxfreq = 0.25/self.delta # half nyquest

        if xmajor_every is None:
            xmajor_every = round(self.distance.max()/4.)
        if xminor_every is None:
            xminor_every = xmajor_every/2.

        xmajorLocator = MultipleLocator(xmajor_every)
        xminorLocator = MultipleLocator(xminor_every)

        ax.set_xlim(self.distance[0], self.distance[-1])
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.set_xlabel('Distance, km')

        ax.set_yscale(freq_scale)
        ax.set_ylim(maxfreq, self.frequency[1])
        ax.set_ylabel('Frequency, Hz')

        if title:
            ax.set_title(title, y=1.03)

        if save is not False:
            fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)
        else:
            if ax is None: fig.show()

        return return_array

def read_seismograms(files, fft=True, cutim=None):
    """
    """

    seismograms = Seismograms()

    if type(files) is str:
        tr = obspy.read(files, format='sac')[0]
        signals = tr.data
    else:
        signals = []
        for f in files:
            tr = obspy.read(f, format='sac')[0]
            signals.append(tr.data)
        signals = np.asarray(signals)

    seismograms.signals = signals
    seismograms.npts = tr.stats.npts
    seismograms.delta = tr.stats.delta
    seismograms.time = tr.times()

    if cutim is not None:
        start, stop = np.array(cutim)/tr.stats.delta
        if type(files) is str:
            seismograms.signals = seismograms.signals[:,int(start):int(stop+1)]
        else:
            seismograms.signals = seismograms.signals[:,int(start):int(stop+1)]
        seismograms.time = seismograms.time[int(start):int(stop+1)]
        seismograms.npts = seismograms.time.size

    # compute the fft
    if fft:
        calc_fft(seismograms)

    return seismograms


def calc_fft(Seismogram_object):
    """
    """
    seismograms = Seismogram_object

    seismograms.frequency = np.fft.rfftfreq(seismograms.npts, seismograms.delta)
    try:
        amplitudes = np.abs(np.fft.rfft(seismograms.signals))/(0.5*seismograms.npts)
    except ValueError:
        amplitudes = []
        for signal in seismograms.signals:
            amplitudes.append(np.abs(np.fft.rfft(signal))/(0.5*seismograms.npts))
        amplitudes = np.asarray(amplitudes)

    seismograms.amplitudes = amplitudes

    return seismograms


class Seismograms(object):
    """
    """

    def __init__(self):
        pass


    def plot_seismogram(self, size=(4, 4), ax=None, colors='k',
                        cut=None, ylim=None,
                        xmajor_every=None, xminor_every=None,
                        ymajor_every=None, yminor_every=None,
                        names=None, title=None, save=False, **kwargs):
        """
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=size)

        if len(self.signals.shape) > 1:
            if colors is 'k':
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i, signal in enumerate(self.signals):
                j = i
                while i > 6:
                    i = i-7
                if names:
                    name = names[j]
                else:
                    name = names
                ax.plot(self.time, signal, color=colors[i], label=name)
        else:
            ax.plot(self.time, self.signals, color=colors, label=names)

        if names:
            ax.legend(loc=0, ncol=2)

        if xmajor_every is None:
            xmajor_every = round(self.time.max()/4.)
        if xminor_every is None:
            xminor_every = xmajor_every/2.
        if ymajor_every is None:
            ymajor_every = self.signals.max()/4.
        if yminor_every is None:
            yminor_every = ymajor_every/2.

        xmajorLocator = MultipleLocator(xmajor_every)
        xminorLocator = MultipleLocator(xminor_every)

        ax.set_xlabel('Time, s')
        if cut:
            ax.set_xlim(cut[0], cut[1])
        else:
            ax.set_xlim(0, self.time[-1])
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.xaxis.set_minor_locator(xminorLocator)

        ymajorLocator = MultipleLocator(ymajor_every)
        yminorLocator = MultipleLocator(yminor_every)

        ax.set_ylabel('Ground velocity, m/s')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)

        if title:
            ax.set_title(title, y=1.03)

        if save is not False:
             fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)
        else:
            if ax is None: fig.show()

        return ax

    def plot_fft(self, size=(4,4), ax=None, colors='k',
            cut=None,  ylim=None,
            names=None, title=None, save=False, **kwargs):
        """
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=size)

        if len(self.amplitudes.shape) > 1:
            if colors is 'k':
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i,amp in enumerate(self.amplitudes):
                j = i
                while i > 6:
                    i = i-7
                if names:
                    name = names[j]
                else:
                    name = names
                ax.plot(self.frequency, amp, color=colors[i], label=name)
        else:
            ax.plot(self.frequency, self.amplitudes, color=colors, label=names)

        if names:
            ax.legend(loc=0)

        ax.set_xlabel('Frequency, Hz')
        ax.set_ylabel('Spectral ground velocity, m/s')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if cut:
            ax.set_xlim(cut[0], cut[1])
        else:
            ax.set_xlim(0.5*self.frequency.min(), 0.25/self.delta)

        if ylim:
            ax.set_ylim(ylim[0], ylim[1])

        if title:
            ax.set_title(title, y=1.03)

        if save is not False:
             fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)
        else:
            if ax is None: fig.show()

        return ax

    def plot_spectral_ratio(self, size=(4,4), ax=None,
            cut=None,  ylim=None, colors=None, ref_index=-1,
            names=None, title=None, save=False, **kwargs):
        """
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=size)

        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        reference = self.amplitudes[ref_index]
        for i,amp in enumerate(self.amplitudes):
            j = i
            while i > 6:
                i = i-7
            if names:
                name = names[j]
            else:
                name = names
            ax.plot(self.frequency, amp/reference, color=colors[i], label=name)

        if names:
            ax.legend(loc=0)

        ax.set_xlabel('Frequency, Hz')
        ax.set_ylabel('Spectral amplification ratio')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if cut:
            ax.set_xlim(cut[0], cut[1])
        else:
            ax.set_xlim(0.5*self.frequency.min(), 0.25/self.delta)

        if ylim:
            ax.set_ylim(ylim[0], ylim[1])

        if title:
            ax.set_title(title, y=1.03)

        if save is not False:
             fig.savefig(save, dpi=720, bbox_inches='tight', transparent=True)
        else:
            if ax is None: fig.show()

        return ax
