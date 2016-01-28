"""
- image.py -

Module to handle WPP and SW4 images of Maps or Cross-Sections

By: Omri Volk & Shahar Shani-Kadmiel, June 2015, kadmiel@post.bgu.ac.il

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from seispy.header import (
    SW4_IMAGE_HEADER_DTYPE, SW4_PATCH_HEADER_DTYPE, SW4_IMAGE_PLANE,
    SW4_IMAGE_MODE_DISPLACEMENT, SW4_IMAGE_MODE_VELOCITY, SW4_IMAGE_PRECISION)
from seispy.plotting import set_matplotlib_rc_params


set_matplotlib_rc_params()


class Image(object):
    """
    A class to hold WPP or SW4 image files
    """
    def __init__(self, source_time_function_type="displacement"):
        self.filename = None
        self.number_of_patches = 1
        self.precision         = 4
        self.cycle             = 0
        self.time              = 0
        self.min               = 0
        self.max               = 0
        self.std               = 0
        self.rms               = 0
        self.patches = []
        # set mode code mapping, depending on the type of source time function
        if source_time_function_type == "displacement":
            self._mode_dict = SW4_IMAGE_MODE_DISPLACEMENT
        elif source_time_function_type == "velocity":
            self._mode_dict = SW4_IMAGE_MODE_VELOCITY
        else:
            msg = ("Unrecognized 'source_time_function_type': '{}'")
            msg = msg.format(source_time_function_type)
            raise ValueError(msg)

    def _readSW4hdr(self, f):
        """
        Read SW4 header information and store it in an Image object
        """
        header = np.fromfile(f, SW4_IMAGE_HEADER_DTYPE, 1)[0]
        (self.precision,
         self.number_of_patches,
         self.time,
         self._plane,
         self.coordinate,
         self._mode,
         self.gridinfo,
         self.creation_time) = header

    @property
    def type(self):
        if self._plane in (0, 1):
            return 'cross-section'
        elif self._plane == 2:
            return 'map'

    @property
    def quantity_name(self):
        return self._mode_dict[self._mode]['name']

    @property
    def quantity_symbol(self):
        return self._mode_dict[self._mode]['symbol']

    @property
    def quantity_unit(self):
        return self._mode_dict[self._mode]['unit']

    def _readSW4patches(self, f):
        """
        Read SW4 patch data and store it in a list of Patch objects
        under Image.patches
        """
        patch_info = np.fromfile(f,SW4_PATCH_HEADER_DTYPE,self.number_of_patches)
        for i,item in enumerate(patch_info):
            patch = Patch()
            patch._image = self
            patch.number = i
            (patch.h,
             patch.zmin,
             patch.ib,
             patch.ni,
             patch.jb,
             patch.nj) = item
            data = np.fromfile(f, self.precision, patch.ni*patch.nj)
            patch.data = data.reshape(patch.nj, patch.ni)

            if self._plane in (0, 1):
                patch.extent = (
                    0 - (patch.h / 2.0),
                    (patch.ni - 1) * patch.h + (patch.h / 2.0),
                    patch.zmin - (patch.h / 2.0),
                    patch.zmin + (patch.nj - 1) * patch.h + (patch.h / 2.0))
            elif self._plane == 2:
                patch.data = patch.data.T
                patch.extent = (
                    0 - (patch.h / 2.0),
                    (patch.nj - 1) * patch.h + (patch.h / 2.0),
                    0 - (patch.h / 2.0),
                    (patch.ni - 1) * patch.h + (patch.h / 2.0))
            patch.min    = data.min()
            patch.max    = data.max()
            patch.std    = data.std()
            patch.rms    = np.sqrt(np.mean(data**2))
            self.patches.append(patch)


class Patch(object):
    """
    A class to hold WPP or SW4 patch data
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
        self._image = None  # link back to the image this patch belongs to

    def plot(self, ax=None, vmin='min', vmax='max', colorbar=True,
             **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_aspect(1)

        if vmax is 'max':
            vmax = self.max
        elif type(vmax) is str:
            try:
                factor = float(vmax)
                vmax = factor*self.rms
                if self.min < 0:
                    vmin = -vmax

            except ValueError:
                print ('Warning! keyword vmax=$s in not understood...\n' %vmax,
                       'Setting to max')
                vmax = self.max

        if vmin is 'min':
            vmin = self.min

        if vmin > self.min and vmax < self.max:
            extend = 'both'
        elif vmin == self.min and vmax == self.max:
            extend = 'neither'
        elif vmin > self.min:
            extend = 'min'
        else:# vmax < self.max:
            extend = 'max'

        print vmin, vmax
        im = ax.imshow(self.data, extent=self.extent, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", **kwargs)
        # invert Z axis if not a map view
        print(self._image._plane)
        if self._image._plane in (0, 1):
            ax.invert_yaxis()
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cb = plt.colorbar(im, cax=cax,
                              extend=extend,
                              label=colorbar if type(colorbar) is str else '')
        else:
            cb = None

        if self._image._plane == 0:
            xlabel = "Y"
            ylabel = "Z"
        elif self._image._plane == 1:
            xlabel = "X"
            ylabel = "Z"
        elif self._image._plane == 2:
            xlabel = "Y"
            ylabel = "X"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        try:
            return fig, ax, cb
        except NameError:
            return cb


def read(filename='random', source_time_function_type="displacement",
         verbose=False):
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
    image = Image(source_time_function_type=source_time_function_type)
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
    elif filename is None:
        pass
    else:
        (name, image.cycle, plane,
         coordinate, mode, is_SW4) = parse_filename(filename)

        if is_SW4:
            with open(image.filename,'rb') as f:
                image._readSW4hdr(f)
                image.precision = SW4_IMAGE_PRECISION[image.precision]
                image.plane = SW4_IMAGE_PLANE[image._plane]
                image._readSW4patches(f)
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
