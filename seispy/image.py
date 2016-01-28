"""
- image.py -

Module to handle WPP and SW4 images of Maps or Cross-Sections

By: Omri Volk & Shahar Shani-Kadmiel, June 2015, kadmiel@post.bgu.ac.il

"""
import warnings

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

    def _read_header(self, f):
        """
        Read SW4 header information and store it in an Image object
        """
        header = np.fromfile(f, SW4_IMAGE_HEADER_DTYPE, 1)[0]
        (self._precision,
         self._number_of_patches,
         self.time,
         self._plane,
         self.coordinate,
         self._mode,
         self.gridinfo,
         self.creation_time) = header

    def _read_patches(self, f):
        """
        Read SW4 patch data and store it in a list of Patch objects
        under Image.patches
        """
        patch_info = np.fromfile(f,SW4_PATCH_HEADER_DTYPE,self._number_of_patches)
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

    def plot(self, patches=None, *args, **kwargs):
        """
        Plot all (or specific) patches in Image.

        >>> my_image.plot()  # plots all patches
        >>> my_image.plot(patches=[0, 2])  # plots first and third patch
        """
        if patches is None:
            for patch in self.patches:
                patch.plot(*args, **kwargs)
        else:
            for i in patches:
                self.patches[i].plot(*args, **kwargs)

    @property
    def is_cross_section(self):
        if self._plane in (0, 1):
            return True
        elif self._plane == 2:
            return False

    @property
    def number_of_patches(self):
        return len(self.patches)

    @property
    def precision(self):
        return SW4_IMAGE_PRECISION[self._precision]

    @property
    def plane(self):
        return SW4_IMAGE_PLANE[self._plane]

    @property
    def type(self):
        if self.is_cross_section:
            return 'cross-section'
        else:
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


class Patch(object):
    """
    A class to hold WPP or SW4 patch data
    """

    def __init__(self):
        self._image = None  # link back to the image this patch belongs to

    def plot(self, ax=None, vmin='min', vmax='max', colorbar=True,
             colorbar_label=None, cmap=None, **kwargs):

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

        im = ax.imshow(self.data, extent=self.extent, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", cmap=cmap,
                       **kwargs)
        # invert Z axis if not a map view
        if self._image.is_cross_section:
            ax.invert_yaxis()
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            if not colorbar_label:
                colorbar_label = "{name} [{unit}]".format(
                    name=self._image.quantity_name,
                    unit=self._image.quantity_unit)
            cb = plt.colorbar(im, cax=cax, extend=extend, label=colorbar_label)
            # invert Z axis for cross-section plots and certain quantities that
            # usually increase with depths
            if self._image.is_cross_section and \
                    self._image._mode in (4, 7, 8):
                cax.invert_yaxis()
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


def read_SW4_image(filename='random', source_time_function_type="displacement",
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

    if filename is 'random':  # generate random data and populate the objects
        image = _create_random_SW4_image(
            source_time_function_type=source_time_function_type)
    elif filename is None:
        pass
    else:
        if not filename.endswith('.sw4img'):
            msg = ("Using 'read_SW4_image()' on file with uncommon file "
                   "extension: '{}'.").format(filename)
            warnings.warn(msg)
        with open(image.filename, 'rb') as f:
            image._read_header(f)
            image._read_patches(f)
    return image


def _create_random_SW4_image(source_time_function_type="displacement"):
    """
    """
    image = Image(source_time_function_type=source_time_function_type)
    image.filename = None
    image._number_of_patches = 1
    image._precision = 4
    image.cycle = 0
    image.time = 0
    image.min = 0
    image.max = 0
    image.std = 0
    image.rms = 0
    image.patches = [_create_random_SW4_patch()]
    return image


def _create_random_SW4_patch():
    patch = Patch()
    patch.ni = 100
    patch.nj = 200
    patch.data = 2 * (np.random.rand(patch.ni, patch.nj) - 0.5)
    patch.number = 0
    patch.h = 100.0
    patch.zmin = 0
    patch.extent = (0, patch.nj * patch.h, patch.zmin + patch.ni * patch.h,
                    patch.zmin)
    patch.min = patch.data.min()
    patch.max = patch.data.max()
    patch.std = patch.data.std()
    patch.rms = np.sqrt(np.mean(patch.data**2))
    patch.ib = 1
    patch.jb = 1
    return patch
