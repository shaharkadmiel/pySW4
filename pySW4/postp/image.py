# -*- coding: utf-8 -*-
"""
Python module to handle SW4 images of Maps or Cross-Sections.

.. module:: image

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:copyright:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

    Omry Volk (omryv@post.bgu.ac.il)

    Tobias Megies (megies@geophysik.uni-muenchen.de)

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

from warnings import warn
import inspect

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

from ..sw4_metadata import Inputfile
from ..headers import (
    IMAGE_HEADER_DTYPE, PATCH_HEADER_DTYPE, IMAGE_PLANE,
    IMAGE_MODE_DISPLACEMENT, IMAGE_MODE_VELOCITY, IMAGE_PRECISION,
    STF)

import copy


class Image():
    """
    A class to hold SW4 image files.

    Initialize an empty Image object, preferentially specifying the
    input file used to run the simulation.

    Parameters
    ----------
    input_file : str or :class:`~obspy.core.util.attribdict.AttribDict`
        Input file (already parsed or filename) used to compute the
        image output.

    stf : {'displacement', 'velocity'}
        Only needed if no metadata from original input_file is used.
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
        path_effects.Stroke(linewidth=1.5 + 1, foreground='w'),
        path_effects.Normal()]

    def __init__(self, input_file=None, stf=None):
        self.patches = []
        if type(input_file) in (str, Inputfile):
            self.input_file = Inputfile(input_file)
            if not self.input_file:
                warn('File {} was not parsed and might not '
                     'be an SW4 inputfile'.format(input_file))
                self.input_file = None
        else:
            self.input_file = None

        try:
            stf_ = STF[self.input_file.source[0].type].type
        except AttributeError:
            stf_ = None
        if (stf and stf_ and
                stf != stf_):
            msg = ('Overriding user specified source time function '
                   'type ({}) with the one found in input file '
                   '({}).')
            warn(msg.format(stf, stf_))
            self.stf = stf_
        else:
            self.stf = stf

        # set mode code mapping, depending on the type of
        # source time function
        if self.stf == 'displacement':
            self._mode_dict = IMAGE_MODE_DISPLACEMENT
        elif self.stf == 'velocity':
            self._mode_dict = IMAGE_MODE_VELOCITY
        else:
            msg = ("Unrecognized 'stf': '{}'")
            raise ValueError(msg.format(self.stf))
        self.filename = None
        self._precision = None
        self.number_of_patches = None
        self.time = None
        self._plane = None
        self.coordinate = None
        self._mode = None
        self.gridinfo = None
        self.extent = None
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
         self.number_of_patches,
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
            f, PATCH_HEADER_DTYPE, self.number_of_patches)

        for i, header in enumerate(patch_info):
            patch = Patch(number=i, image=self)
            patch._set_header(header)
            data = np.fromfile(f, self.precision, patch.ni * patch.nj)
            data = data.reshape(patch.nj, patch.ni)
            patch._set_data(data)
            if (self.gridinfo and
                    patch.number == self.number_of_patches - 1):
                patch.is_curvilinear = True
            else:
                patch.is_curvilinear = False

            self.patches.append(patch)

        if self.gridinfo == 1:
            self._read_curvilinear_grid(f)

    def _read_curvilinear_grid(self, f):
        """
        Read the last bit of the SW4 image file in case a curvilinear
        grid is found.

        Parameters
        ----------
        f : file
            Open file handle of SW4 image file (at correct position).
        """
        grid = Patch(number='curvilinear grid', image=self)

        # get data from the corresponding image patch
        patch = self.patches[-1]

        grid.h = patch.h
        grid.ib = patch.ib
        grid.ni = patch.ni
        grid.jb = patch.jb
        grid.nj = patch.nj

        # read data from file
        data = np.fromfile(f, self.precision, grid.ni * grid.nj)
        data = data.reshape(grid.nj, grid.ni)

        zmin = data.min()
        grid.zmin = zmin
        grid._set_data(data)

        # update curvilinear patch and grid extent
        extent = list(grid.extent)
        extent[2] = zmin - (grid.h / 2.0)
        extent[3] = grid._max + (grid.h / 2.0)
        extent = tuple(extent)
        grid.extent = extent

        self.patches[-1].zmin = zmin
        self.patches[-1].extent = extent

        self.curvilinear_grid_patch = grid

    def _calc_global_min_max(self):
        """
        Calculate ``min``, ``max``, ``rms``, and ``ptp``.
        """
        _max = []
        _min = []
        _rms = []
        for patch in self.patches:
            _max += [patch._max]
            _min += [patch._min]
            _rms += [patch._rms]
        self._max = max(_max)
        self._min = min(_min)
        self._rms = max(_rms)
        self._ptp = self._max - self._min

        if self.is_cross_section:
            x1, x2 = self.patches[-1].extent[:2]
            y1 = min([patch.extent[2] for patch in self.patches])
            y2 = max([patch.extent[3] for patch in self.patches])
            self.extent = (x1, x2, y1, y2)
        else:
            self.extent = self.patches[-1].extent

    def plot(self, patches=None, ax=None, vmin='min', vmax='max',
             colorbar=True, cmap=None, interpolation='nearest',
             origin='lower', projection_distance=np.inf, **kwargs):
        """
        Plot all (or specific) patches in :class:`~.Image`.

        Parameters
        ----------
        patches : list of int
            Patches to plot

        Other Parameters
        ----------------
        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes.

        vmin : float
            Manually set minimum of color scale.

        vmax : str or float
            Used to clip the coloring of the data at the set value.
            Default is 'max' which shows all the data. If  ``float``,
            the colorscale saturates at the given value. Finally, if a
            string is passed (other than 'max'), it is casted to float
            and used as an ``rms`` multiplier. For instance, if
            ``vmax='3'``, clipping is done at 3.0\*rms of the data.

        colorbar : bool or str
            If ``colorbar`` is a string, that string is used to override
            the automatic label chosen based on the image header. To
            Supress plotting of the colorbar set to ``False``.

        cmap : :class:`~matplotlib.colors.Colormap`
            Colormap for the plot

        interpolation : str
            Acceptable values are 'none', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
            'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
            'bessel', 'mitchell', 'sinc', 'lanczos'.

        origin : str
            Places the origin at the 'lower' (default)
            or 'upper' left corner of the plot.

        projection_distance : float
            Threshold distance from the plane coordinate to include
            symbols of stations and sources. These are orthogonally
            *projected* onto the plotted 2D plane. By default everything
            is included but this can cause too many symbols to be
            plotted, obscuring the image.

        Examples
        --------
        >>> my_image.plot()  # plots all patches
        >>> my_image.plot(patches=[0, 2])  # plots first and third patch
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        # set vmin, vmax
        if vmax is None or vmax is 'max':
            clip = self._max
        elif type(vmax) in [float, int]:
            clip = vmax
        else:
            clip = float(vmax) * self._rms

        if vmin is None or vmin is 'min':
            vmin = self._min
        elif type(vmin) in [int, float]:
            pass
        elif vmin in [None, False]:
            vmin = -clip

        vmax = clip

        # plot patches
        if patches is None:
            for patch in self.patches:
                im = patch.plot(
                    ax=ax, vmin=vmin, vmax=vmax, colorbar=False,
                    cmap=cmap, interpolation=interpolation, origin=origin,
                    **kwargs)
        else:
            for i in patches:
                im = self.patches[i].plot(
                    ax=ax, vmin=vmin, vmax=vmax, colorbar=False,
                    cmap=cmap, interpolation=interpolation, origin=origin,
                    **kwargs)

        # plot the colorbar
        if colorbar:
            # set colorbar extend mode
            if vmin > self._min and vmax < self._max:
                extend = 'both'
            elif vmin > self._min:
                extend = 'min'
            elif vmax < self._max:
                extend = 'max'
            else:
                extend = 'neither'

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            if type(colorbar) is str:
                colorbar_label = colorbar
            else:
                colorbar_label = "{name} [{unit}]".format(
                    name=self.quantity_name,
                    unit=self.quantity_unit)
            cb = plt.colorbar(im, cax=cax, extend=extend,
                              label=colorbar_label)
            # invert Z axis for cross-section plots and certain
            # quantities that usually increase with depths
            if (self.is_cross_section and
                    self._mode in (4, 7, 8)):
                cax.invert_yaxis()
        else:
            cb = None

        # labels
        if self._plane == 0:
            xlabel = "Y"
            ylabel = "Z"
        elif self._plane == 1:
            xlabel = "X"
            ylabel = "Z"
        elif self._plane == 2:
            xlabel = "Y"
            ylabel = "X"
        ax.set_xlabel(xlabel + ' [m]')
        ax.set_ylabel(ylabel + ' [m]')

        title = self.filename.rsplit('.', 1)[0]
        if len(title) > 40:
            title = '...' + title[-40:]

        ax.set_title("{}\n{}={}  t={:.2f} seconds".format(
            title, self.plane, self.coordinate,
            self.time), y=1.03, fontsize=12)

        # plot receivers, source etc.
        if self.input_file:
            try:  # get surface elevation to correct depth coordinate
                xi = self.curvilinear_grid_patch.x
                elev = self.curvilinear_grid_patch.data[0]
            except AttributeError:  # no curvilinear grid... no correction
                xi = None
                elev = None

            for key, kwargs in self.MPL_SCATTER_PROPERTIES.items():
                coordinates = self.input_file.get_coordinates(
                    key, xi, elev,
                    self._plane, self.coordinate, projection_distance)

                if not coordinates:
                    continue
                x, y = coordinates
                collection = ax.scatter(
                    x, y, **kwargs)
                collection.set_path_effects(
                    self.MPL_SCATTER_PATH_EFFECTS)
        # reset axes limits to data, they sometimes get changed by the
        # scatter plot
        ax.axis(self.extent)

        # invert Z axis if not a map view
        if self.is_cross_section:
            ax.invert_yaxis()
        return fig, ax, cb

    def get_source_coordinates(self):
        try:
            return self._get_plot_coordinates_from_input("source")
        except KeyError:
            return None

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
    def _cmap_type(self):
        try:
            return self._mode_dict[self._mode]['cmap_type']
        except KeyError:
            return None

    @property
    def precision(self):
        try:
            return IMAGE_PRECISION[self._precision]
        except KeyError:
            return None

    @property
    def plane(self):
        try:
            return IMAGE_PLANE[self._plane]
        except KeyError:
            return None

    @property
    def type(self):
        if self.is_cross_section:
            return 'cross-section'
        else:
            return 'map'

    @property
    def quantity_name(self):
        try:
            return self._mode_dict[self._mode]['name']
        except KeyError:
            return None

    @property
    def quantity_altname(self):
        try:
            return self._mode_dict[self._mode]['altname']
        except KeyError:
            return None

    @property
    def quantity_symbol(self):
        try:
            return self._mode_dict[self._mode]['symbol']
        except KeyError:
            return None

    @property
    def quantity_altsymbol(self):
        try:
            return self._mode_dict[self._mode]['altsymbol']
        except KeyError:
            return None

    @property
    def quantity_unit(self):
        try:
            return self._mode_dict[self._mode]['unit']
        except KeyError:
            return None

    def __str__(self):
        string = (
            '        Image information :\n'
            '        ----------------- :\n'
            '                 Filename : {}\n'
            '                Precision : {}\n'
            '        Number of patches : {}\n'
            '                  Time, s : {}\n'
            '                    Plane : {}\n'
            '               Coordinate : {}\n'
            '                     Mode : {}, {}\n'
            'Curvilinear grid included : {}\n'
            '             Image extent : {}\n'
            '            Creation time : {}\n'
        ).format(
            str(self.filename).rsplit('/', 1)[-1], self.precision,
            self.number_of_patches, self.time,
            self.plane, self.coordinate, self.quantity_symbol,
            self.quantity_unit, True if self.gridinfo else False,
            self.extent, self.creation_time
        )

        return string

    def copy(self):
        """
        Return a deepcopy of `self`.
        """
        return copy.deepcopy(self)


class Patch():
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
        self._min = data.min()
        self._max = data.max()
        self._std = data.std()
        self._rms = np.sqrt(np.mean(data**2))
        self.shape = data.shape

    def plot(self, ax=None, vmin=None, vmax=None, colorbar=True, cmap=None,
             interpolation='nearest', origin='lower', **kwargs):
        """
        Plot patch.

        .. note:: Should not really be used directly by the user but
                  rather called by the :meth:`~.Image.plot` method.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes.

        vmin : float
            Manually set minimum of color scale.

        vmax : float
            Manually set maximum of color scale.

        colorbar : bool or str
            If ``colorbar`` is a string, that string is used to override
            the automatic label chosen based on the image header. To
            Supress plotting of the colorbar set to ``False``.

        cmap : :class:`~matplotlib.colors.Colormap`
            Colormap for the plot

        interpolation : str
            Acceptable values are 'none', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
            'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
            'bessel', 'mitchell', 'sinc', 'lanczos'.

        origin : str
            Places the origin at the 'lower' (default)
            or 'upper' left corner of the plot.

        Other Parameters
        ----------------
        kwargs : :func:`~matplotlib.pyplot.imshow`
        """
        caller = inspect.stack()[1][3]  # find out who made the call

        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = None

        if not cmap:
            cmap = self._image.CMAP[self._image._cmap_type]

        ax.set_aspect(1)

        if self.is_curvilinear:
            y = self._image.curvilinear_grid_patch.data
            x = (np.ones_like(y) * self.x)

            im = ax.pcolormesh(x, y, self.data,
                               shading='flat', edgecolors='None',
                               vmin=vmin, vmax=vmax, cmap=cmap)
            ax.axis(self.extent)
        elif not self.is_curvilinear:
            im = ax.imshow(self.data, extent=self.extent,
                           vmin=vmin, vmax=vmax, origin=origin,
                           interpolation=interpolation, cmap=cmap, **kwargs)

        # invert Z axis if not a map view
        if self._image.is_cross_section:
            ax.invert_yaxis()

        if caller is not 'plot':  # do this only if the user calls
            if colorbar:
                if vmin > self._min and vmax < self._max:
                    extend = 'both'
                elif vmin > self._min:
                    extend = 'min'
                elif vmax < self._max:
                    extend = 'max'
                else:
                    extend = 'neither'

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                if type(colorbar) is str:
                    colorbar_label = colorbar
                else:
                    colorbar_label = "{name} [{unit}]".format(
                        name=self._image.quantity_name,
                        unit=self._image.quantity_unit)
                cb = plt.colorbar(im, cax=cax, extend=extend,
                                  label=colorbar_label)
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
                title = ('...' + title[-40:]
                         + ' : patch no. {}'.format(self.number))

            ax.set_title("{}\n{}={}  t={:.2f} seconds".format(
                title, self._image.plane, self._image.coordinate,
                self._image.time), y=1.03, fontsize=12)
            return fig, ax, cb

        elif caller is 'plot':  # do this only if Image.plot() calls
            return im

    def __str__(self):
        string = (
            'Patch information :\n'
            '----------------- :\n'
            '           Number : {}\n'
            '       Spacing, m : {}\n'
            '          Zmin, m : {}\n'
            '           Extent : {}\n'
            '               ni : {}\n'
            '               nj : {}\n'
            '              Max : {}\n'
            '              Min : {}\n'
            '              STD : {}\n'
            '              RMS : {}\n'
        ).format(
            self.number, self.h, self.zmin, self.extent, self.ni, self.nj,
            self._max, self._min, self._std, self._rms
        )
        return string

    @property
    def x(self):
        return np.linspace(*self.extent[:2], num=self.ni)

    def copy(self):
        """
        Return a deepcopy of `self`.
        """
        return copy.deepcopy(self)


def read_image(filename='random', input_file=None,
               stf="displacement", verbose=False):
    """
    Read image data, cross-section or map into a
    :class:`.Image` object.

    Parameters
    ----------
    filename : str
        If no filename is passed, by default, a random image is
        generated. if filename is ``None``, an empty :class:`.Image`
        object is returned.

    input_file : str or AttribDict
        Input file (already parsed or filename) used to compute the
        image output.

    stf : str
        `'displacement'` or `'velocity'`. Only needed if no metadata
        from original input_file is used.

    verbose : bool
        If set to ``True``, print some information while reading the
        file.

    Returns
    -------
    :class:`.Image`
        An :class:`~.Image` object with a list of :class:`~.Patch`
        objects.
    """
    image = Image(stf=stf,
                  input_file=input_file)
    image.filename = filename

    if filename is 'random':  # generate random data, populate objects
        image = _create_random_image(
            stf=stf)
    elif filename is None:
        pass
    else:
        if not filename.endswith('.sw4img'):
            msg = ("Using 'read_image()' on file with uncommon file "
                   "extension: '{}'.").format(filename)
            warn(msg)
        with open(image.filename, 'rb') as f:
            image._read_header(f)
            image._read_patches(f)
        image._calc_global_min_max()
    return image


def _create_random_image(stf="displacement"):
    image = Image(stf=stf)
    image.filename = None
    image.number_of_patches = 1
    image._precision = 4
    image.cycle = 0
    image.time = 0
    image._min = 0
    image._max = 0
    image._rms = 0
    image._ptp = 0
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
    patch._min = patch.data.min()
    patch._max = patch.data.max()
    patch._rms = np.sqrt(np.mean(patch.data**2))
    patch.ib = 1
    patch.jb = 1
    return patch
