# -*- coding: utf-8 -*-
"""
Python module to handle SW4 images of Maps or Cross-Sections.

.. module:: image

:author:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:copyright:
    Shahar Shani-Kadmiel (kadmiel@post.bgu.ac.il)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy.core.util import AttribDict
try:
    from obspy.imaging.cm import obspy_divergent as cmap_divergent
    from obspy.imaging.cm import obspy_divergent_r as cmap_divergent_r
    from obspy.imaging.cm import obspy_sequential as cmap_sequential
    from obspy.imaging.cm import obspy_sequential_r as cmap_sequential_r
except ImportError:
    cmap_divergent = None
    cmap_divergent_r = None
    cmap_sequential = None
    cmap_sequential_r = None

from .config import read_input_file
from .header import (
    IMAGE_HEADER_DTYPE, PATCH_HEADER_DTYPE, IMAGE_PLANE,
    IMAGE_MODE_DISPLACEMENT, IMAGE_MODE_VELOCITY, IMAGE_PRECISION,
    SOURCE_TIME_FUNCTION_TYPE)


class Image(object):
    """
    A class to hold SW4 image files.

    Initialize an empty Image object, preferentially specifying the
    config (file) used to run the simulation.

    Parameters
    ----------
    config : str or :class:`~obspy.core.util.attribdict.AttribDict`
        Configuration (already parsed or filename) used to compute the
        image output.

    source_time_function_type : str
        `'displacement'` or `'velocity'`. Only needed if no metadata
        from original config is used.
    """
    CMAP = {"divergent"    : cmap_divergent,
            "divergent_r"  : cmap_divergent_r,
            "sequential"   : cmap_sequential,
            "sequential_r" : cmap_sequential_r}
    MPL_SCATTER_PROPERTIES = {
        "source" : {"s"          : 200,
                    "marker"     : "*",
                    "edgecolors" : "k",
                    "facecolors" : "",
                    "alpha"      : 1,
                    "linewidths" : 1.5},
        "rec"    : {"s"          : 200,
                    "marker"     : "v",
                    "edgecolors" : "k",
                    "facecolors" : "",
                    "alpha"      : 1,
                    "linewidths" : 1.5}
        }
    MPL_SCATTER_PATH_EFFECTS = [
        path_effects.Stroke(linewidth=1.5 + 0.7, foreground='w'),
        path_effects.Normal()]

    def __init__(self, config=None, source_time_function_type=None):

        self.patches = []
        if config is not None and not isinstance(config, AttribDict):
            config = read_input_file(config)
        self._config = config
        if config:
            source_time_function_type_ = self.source_time_function_type
            if source_time_function_type and source_time_function_type_ and \
                    source_time_function_type != source_time_function_type_:
                msg = ("Overriding user specified source time function "
                       "type ({}) with the one found in configuration file "
                       "({}).").format(source_time_function_type,
                                       source_time_function_type_)
                warnings.warn(msg)
            source_time_function_type = source_time_function_type_
        # set mode code mapping, depending on the type of source time function
        if source_time_function_type == "displacement":
            self._mode_dict = IMAGE_MODE_DISPLACEMENT
        elif source_time_function_type == "velocity":
            self._mode_dict = IMAGE_MODE_VELOCITY
        else:
            msg = ("Unrecognized 'source_time_function_type': '{}'")
            msg = msg.format(source_time_function_type)
            raise ValueError(msg)
        self.filename = None
        self._precision = None
        self._number_of_patches = None
        self.time = None
        self._plane = None
        self.coordinate = None
        self._mode = None
        self.gridinfo = None
        self.creation_time = None

    def _read_header(self, f):
        """
        Read SW4 header information and store it in an :class:`~.Image`
        object.

        Parameters
        ----------
        f : file
            Open file handle of SW4 image file (at correct position).
        """
        header = np.fromfile(f, IMAGE_HEADER_DTYPE, 1)[0]
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
        Read SW4 patch data and store it in a list of :class:`~.Patch`
        objects under :obj:`~.Image.patches`.

        Parameters
        ----------
        f : file
            Open file handle of SW4 image file (at correct position).
        """
        patch_info = np.fromfile(
            f, PATCH_HEADER_DTYPE, self._number_of_patches)

        for i, header in enumerate(patch_info):
            patch = Patch(number=i, image=self)
            patch._set_header(header)
            data = np.fromfile(f, self.precision, patch.ni * patch.nj)
            data = data.reshape(patch.nj, patch.ni)
            patch._set_data(data)
            self.patches.append(patch)

    def plot(self, patches=None, *args, **kwargs):
        """
        Plot all (or specific) patches in :class:`~.Image`.

        Parameters
        ----------
        patches : list of int
            Patches to plot


        .. rubric:: **Other keywoard arguments from**
            :meth:`.Patch.plot` **args/kwargs:**

        Keyword Arguments
        -----------------
        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes.

        vmin : float
            Manually set minimum of color scale.

        vmax : float
            Manually set maximum of color scale.

        colorbar : bool
            Whether to plot colorbar.

        colorbar_label : str
            Label for colorbar.

        cmap : :class:`~matplotlib.colors.Colormap`
            Colormap for the plot

        Example
        -------
        >>> my_image.plot()  # plots all patches
        >>> my_image.plot(patches=[0, 2])  # plots first and third patch
        """
        if patches is None:
            for patch in self.patches:
                patch.plot(*args, **kwargs)
        else:
            for i in patches:
                self.patches[i].plot(*args, **kwargs)

    def _get_plot_coordinates_from_config(self, key):
        """
        Gets coordinates for config keys that have 3D x, y, z values
        (e.g. 'source', 'rec') in 2D plotting coordinates for use in
        :meth:`.plot`.
        """
        if not self._config:
            return None
        items = self._config.get(key, [])
        if not items:
            return None
        x = []
        y = []
        for item in items:
            if self._plane == 0:
                x_ = item.y
                y_ = item.z
            elif self._plane == 1:
                x_ = item.x
                y_ = item.z
            elif self._plane == 2:
                x_ = item.y
                y_ = item.x
            x.append(x_)
            y.append(y_)
        if not x:
            return None
        return x, y

    def get_source_coordinates(self):
        return self._get_plot_coordinates_from_config("source")

    @property
    def is_cross_section(self):
        if self._plane in (0, 1):
            return True
        elif self._plane == 2:
            return False

    @property
    def is_divergent(self):
        if self._cmap_type in ("divergent", "divergent_r"):
            return True
        return False

    @property
    def source_time_function_type(self):
        if not self._config:
            return None
        stf_type = SOURCE_TIME_FUNCTION_TYPE[self._config.source[0].type]
        return {0: "displacement", 1: "velocity"}.get(stf_type, None)

    @property
    def _cmap_type(self):
        return self._mode_dict[self._mode]['cmap_type']

    @property
    def number_of_patches(self):
        return len(self.patches)

    @property
    def precision(self):
        return IMAGE_PRECISION[self._precision]

    @property
    def plane(self):
        return IMAGE_PLANE[self._plane]

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
    def quantity_altname(self):
        return self._mode_dict[self._mode]['altname']

    @property
    def quantity_symbol(self):
        return self._mode_dict[self._mode]['symbol']

    @property
    def quantity_altsymbol(self):
        return self._mode_dict[self._mode]['altsymbol']

    @property
    def quantity_unit(self):
        return self._mode_dict[self._mode]['unit']


class Patch(object):
    """
    A class to hold SW4 patch data.

    Initialize an empty Patch object, preferentially specifying the
    parent :class:`.Image`.

    Parameters
    ----------
    image : :class:`.Image`
        Parent Image object.

    number : int
        Patch index in parent image (starts at `0`).
    """
    def __init__(self, image=None, number=None):

        self.number = number
        self._image = image  # link back to the image this patch belongs to
        self.h = None
        self.zmin = None
        self.ib = None
        self.ni = None
        self.jb = None
        self.nj = None
        self.data = None
        self.extent = None

    def _set_header(self, header):
        """
        Set SW4 patch header information
        """
        (self.h,
         self.zmin,
         self.ib,
         self.ni,
         self.jb,
         self.nj) = header

    def _set_data(self, data):
        self.data = data
        if self._image.is_cross_section:
            self.extent = (
                0 - (self.h / 2.0),
                (self.ni - 1) * self.h + (self.h / 2.0),
                self.zmin - (self.h / 2.0),
                self.zmin + (self.nj - 1) * self.h + (self.h / 2.0))
        else:
            self.data = self.data.T
            self.extent = (
                0 - (self.h / 2.0),
                (self.nj - 1) * self.h + (self.h / 2.0),
                0 - (self.h / 2.0),
                (self.ni - 1) * self.h + (self.h / 2.0))
        self.min = data.min()
        self.max = data.max()
        self.std = data.std()
        self.rms = np.sqrt(np.mean(data**2))
        self.shape = data.shape

    def plot(self, ax=None, vmin=None, vmax=None, colorbar=True,
             colorbar_label=None, cmap=None, **kwargs):
        """
        Plot patch and show plot.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes.

        vmin : float
            Manually set minimum of color scale.

        vmax : float
            Manually set maximum of color scale.

        colorbar : bool
            Whether to plot colorbar.

        colorbar_label : str
            Label for colorbar.

        cmap : :class:`~matplotlib.colors.Colormap`
            Colormap for the plot


        Note
        ----
        For other keywoard arguments (``**kwargs``) see:
        :func:`matplotlib.pyplot.imshow`.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if cmap is None:
            cmap = self._image.CMAP[self._image._cmap_type]

        ax.set_aspect(1)

        # center colormap around zero if image's data is divergent
        if self._image.is_divergent:
            if vmin is None and vmax is None:
                abs_max = max(abs(self.min), abs(self.max))
                vmax = abs_max
                vmin = -abs_max

        if (vmin is not None and self.min < vmin) and \
                (vmax is not None and self.max > vmax):
            extend = 'both'
        elif vmin is not None and self.min < vmin:
            extend = 'min'
        elif vmax is not None and self.max > vmax:
            extend = 'max'
        else:
            extend = 'neither'

        im = ax.imshow(self.data, extent=self.extent, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", cmap=cmap,
                       **kwargs)
        # plot receiver, source etc.
        if self._image._config:
            for key, kwargs in self._image.MPL_SCATTER_PROPERTIES.items():
                coordinates = \
                    self._image._get_plot_coordinates_from_config(key)
                if not coordinates:
                    continue
                x, y = coordinates
                collection = ax.scatter(
                    x, y, **kwargs)
                collection.set_path_effects(
                    self._image.MPL_SCATTER_PATH_EFFECTS)
            # reset axes limits to data, they sometimes get changed by the
            # scatter plot
            ax.set_xlim(*self.extent[:2])
            ax.set_ylim(*self.extent[2:])

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
            # invert Z axis for cross-section plots and certain
            # quantities that usually increase with depths
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
        ax.set_xlabel(xlabel + ' [m]')
        ax.set_ylabel(ylabel + ' [m]')

        # setup the title so that it is not too long
        # I guess for some purposes you would want the
        # entire path to show so we might want to just
        # brake it into lines?
        # For now I am truncating it at -40: chars and
        # getting rid of the .sw4img extention... That's
        # kind of obvious.
        title = self._image.filename.rsplit('.', 1)[0]
        if len(title) > 40:
            title = '...' + title[-40:]

        ax.set_title("{}\n{}={}  t={:.2f} seconds".format(
            title, self._image.plane, self._image.coordinate,
            self._image.time), y=1.03, fontsize=12)

        try:
            return fig, ax, cb
        except NameError:
            return cb


def read_image(filename='random', config=None,
               source_time_function_type="displacement", verbose=False):
    """
    Read image data, cross-section or map into a
    :class:`.Image` object.

    Parameters
    ----------
    filename : str
        If no filename is passed, by default, a random image is
        generated. if filename is ``None``, an empty :class:`.Image`
        object is returned.

    config : str or AttribDict
        Configuration (already parsed or filename) used to compute the
        image output.

    source_time_function_type : str
        `'displacement'` or `'velocity'`. Only needed if no metadata
        from original config is used.

    verbose : bool
        If set to ``True``, print some information while reading the
        file.

    Returns
    -------
    :class:`.Image`
        An :class:`~.Image` object with a list of :class:`~.Patch`
        objects.
    """
    image = Image(source_time_function_type=source_time_function_type,
                  config=config)
    image.filename = filename

    if filename is 'random':  # generate random data, populate objects
        image = _create_random_image(
            source_time_function_type=source_time_function_type)
    elif filename is None:
        pass
    else:
        if not filename.endswith('.sw4img'):
            msg = ("Using 'read_image()' on file with uncommon file "
                   "extension: '{}'.").format(filename)
            warnings.warn(msg)
        with open(image.filename, 'rb') as f:
            image._read_header(f)
            image._read_patches(f)
    return image


def _create_random_image(source_time_function_type="displacement"):
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
    image.patches = [_create_random_patch()]
    return image


def _create_random_patch():
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
