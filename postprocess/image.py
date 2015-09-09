"""
- image.py -

Module to handle WPP and SW4 images of Maps or Cross-Sections

By: Omri Volk & Shahar Shani-Kadmiel, June 2015, kadmiel@post.bgu.ac.il

"""

import os
import numpy as np
from seispy.plotting import patch_plot
from seispy.plotting.dic_and_dtype import *

class Image(object):
    """
    A class to hold WPP or SW4 image files
    """

    def __init__(self):
        self.filename          = None
        self.number_of_patches = 1 # or more
        self.precision         = 4 # or 8
        self.type              = 'cross-section' # or or 'map'
        self.mode              = None # velmag, ux, uy, uz, etc.
        self.unit              = None # m, m/s, kg/cm^3, etc.
        self.cycle             = 0
        self.time              = 0
        self.plane             = 'X' # or Y or Z
        self.min               = 0
        self.max               = 0
        self.std               = 0
        self.rms               = 0

        self.patches = []

class Patch(object):
    """
    A class to hold WW or SW4 patch data
    """

    def __init__(self):
        self.number       = 0
        self.h            = 0
        self.zmin         = 1
        self.ib           = 1
        self.ni           = 0
        self.jb           = 1
        self.nj           = 0
        self.extent       = (0,1,0,1)
        self.data         = None
        self.min          = 0
        self.max          = 0
        self.std          = 0
        self.rms          = 0

    def plot(self, *args, **kwargs):
        """
        """
        return patch_plot(self, *args, **kwargs)


def read(filename='random', verbose=False):
    """
    Read image data, cross-section or map into a SeisPy Image object.

    Params:
    --------

    filename : if no filename is passed, by default, a random image is generated
        if filename is None, an empty Image object is returned.

    verbose : if True, print some information while reading the file.

    Returns:
    ---------

    an Image object with a list of Patch objects
    """

    image = Image()
    image.filename = filename

    if filename is 'random': # generate random data and populate the objects
        patch = Patch()

        ni,nj = 100,200
        h = 100.
        zmin = 0
        data = 2*(np.random.rand(ni,nj)-0.5)

        patch.data   = data
        patch.ni     = ni
        patch.nj     = nj
        patch.number = 0
        patch.h      = h
        patch.zmin   = zmin
        patch.extent = (0,nj*h,zmin+ni*h,zmin)
        patch.min    = data.min()
        patch.max    = data.max()
        patch.std    = data.std()
        patch.rms    = np.sqrt(np.mean(data**2))

        image.patches += [patch]
        return image

    elif filename is None:
        return image

    else:
        (name, image.cycle, plane,
         coordinate, mode, is_SW4) = parse_filename(filename)

    if is_SW4:
        with open(image.filename,'rb') as f:
            readSW4hdr(image, f)
            image.precision = prec_dict[image.precision]
            image.plane = SW4_plane_dict[image.plane]
            image.mode, image.unit = SW4_mode_dict[image.mode]

            position = 61 + 32*image.number_of_patches
            for patch in image.patches:
                readSW4patch(patch, f, image.precision)

        return image


def parse_filename(filename):
    """ This function parses the filename in order to figure out its type.

    Parameters
    -----------
    filename : string

    Returns
    --------
    name, cycle, plane, coordinate, mode, is_SW4

    """

    basename = os.path.basename(filename)
    name, extention = os.path.splitext(basename)
    if extention == '.sw4img':
        name, cycle, plane, mode = name.rsplit('.',3)
        cycle = int(cycle.split('=')[-1])
        plane, coordinate = plane.split('=')
        return name, cycle, plane, coordinate, mode, True
    else:
        name, cycle, plane, mode = basename.rsplit('.',3)
        cycle = int(cycle.split('=')[-1])
        plane, coordinate = plane.split('=')
        return name, cycle, plane, coordinate, mode, False


def readSW4hdr(image, f):
    """
    Read SW4 header information and store it in an Image object
    """

    header = np.fromfile(f, SW4_header_dtype, 1)[0]
    (image.precision,
     image.number_of_patches,
     image.time,
     image.plane,
     image.coordinate,
     image.mode,
     image.gridinfo,
     image.creation_time) = header

    patch_info = np.fromfile(f,SW4_patch_dtype,image.number_of_patches)
    for i,item in enumerate(patch_info):
        patch = Patch()
        patch.number = i
        (patch.h,
         patch.zmin,
         patch.ib,
         patch.ni,
         patch.jb,
         patch.nj) = item
        image.patches += [patch]
    return


def readSW4patch(patch, f, dtype):
    """
    Read SW4 patch data and store it in a list of Patch objects
    under Image.patches
    """

    data = np.fromfile(f, dtype, patch.ni*patch.nj)
    patch.data = data.reshape(patch.nj,patch.ni)

    patch.extent = (0,patch.nj*patch.h,patch.zmin+patch.ni*patch.h,patch.zmin)
    patch.min    = data.min()
    patch.max    = data.max()
    patch.std    = data.std()
    patch.rms    = np.sqrt(np.mean(data**2))
    return

def plot(self,**imshow_kwargs):

    sw4_plot_image(self,**imshow_kwargs)
