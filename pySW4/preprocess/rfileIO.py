# -*- coding: utf-8 -*-
"""
Python module to read and write rfiles.

.. module:: rfileIO

:author:
    Shahar Shani-Kadmiel
    kadmiel@post.bgu.ac.il

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import, print_function, division

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from warnings import warn

flush = sys.stdout.flush()

RFILE_HEADER_DTYPE = np.dtype([
        ('magic'       , 'int32'   ),
        ('precision'   , 'int32'   ),
        ('attenuation' , 'int32'   ),
        ('az'          , 'float64' ),
        ('lon0'        , 'float64' ),
        ('lat0'        , 'float64' ),
        ('mlen'        , 'int32'   )
    ])

BLOCK_HEADER_DTYPE = np.dtype([
        ('hh'  ,  'float64' ),
        ('hv'  ,  'float64' ),
        ('z0'  ,  'float64' ),
        ('nc'  ,  'int32'   ),
        ('ni'  ,  'int32'   ),
        ('nj'  ,  'int32'   ),
        ('nk'  ,  'int32'   )
    ])

DATA_PRECISION = {4: np.float32, 8: np.float64}

COMPONENTS = {'rho' : 0,
              'vp'  : 1,
              'vs'  : 2,
              'qp'  : 3,
              'qs'  : 4}

TITLES_AND_LABELS = {
    0 : {'name'      : 'Density',
         'symbol'    : 'rho',
         'unit'      : 'kg/m^3'},
    1 : {'name'      : 'P Wave Velocity',
         'symbol'    : 'Vp',
         'unit'      : 'm/s'},
    2 : {'name'      : 'S Wave Velocity',
         'symbol'    : 'Vs',
         'unit'      : 'm/s'},
    3 : {'name'      : 'Qp',
         'symbol'    : 'Qp',
         'unit'      : ''},
    4 : {'name'      : 'Qs',
         'symbol'    : 'Qs',
         'unit'      : ''},
    }


def write_hdr(f, magic=1, precision=4, attenuation=1,
              az=0., lon0=33.5, lat0=28.0,
              proj_str=('+proj=utm +zone=36 +datum=WGS84 '
                        '+units=m +no_defs'),
              nb=1):
    """
    Write rfile header.

    Parameters
    ----------
    f : file
        Open file handle in ``'wb'`` mode

    magic : int
        Determine byte ordering in file. Defaults to 1.

    precision : int
        The number of bytes per entry in the data section.
        4 - single precision (default)
        8 - double precision

    attenuation : int
        Indicates whether the visco-elastic attenuation parameters QP
        and QS are included in the data section.
        0 - no visco-elastic attenuation parameters included
        1 - visco-elastic attenuation parameters included (default)

    az : float
        Angle in degrees between North and the positive x-axis.
        Defaults to 0. See the `SW4 User Guide`_.

    lon0 : float
        Longitude of the origin of the data. Defaults to 33.5.

    lat0 : float
        Latitude of the origin of the data. Defaults to 28.0.

    proj_str : str
        Projection string which is read by the Proj4 library if SW4 was
        built with Proj4 support. See the `SW4 User Guide`_ and the
        `Proj4`_ documentation. Defaults to
        '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs'.

    nb : int
        The number of blocks in the data section. Must be > 0.
        Defaults to 1.


    .. _SW4 User Guide:
       https://geodynamics.org/cig/software/sw4/

    .. _Proj4:
       https://trac.osgeo.org/proj/wiki/GenParms
    """

    magic        = np.int32(magic)
    precision    = np.int32(precision)
    attenuation  = np.int32(attenuation)
    az           = np.float64(az)
    lon0         = np.float64(lon0)
    lat0         = np.float64(lat0)
    mlen         = np.int32(len(proj_str))
    nb           = np.int32(nb)

    hdr = [magic, precision, attenuation, az, lon0, lat0, mlen, proj_str, nb]
    for val in hdr:
        f.write(val)
    return


def read_hdr(f):
    """
    Read rfile header.

    Parameters
    ----------
    f : file
        Open file handle in ``'rb'`` mode of the rfile to be read.

    Returns
    -------
    tuple
        A tuple containing rfile header data (9 elements):
            magic, precision, attenuation,
            az, lon0, lat0, mlen, proj_str, nb


    .. rubric:: **Description of header data returned**

    ``magic``
        Determine byte ordering in file.

    ``precision``
        The number of bytes per entry in the data section.
        4 - single precision
        8 - double precision

    ``attenuation``
        Indicates whether the visco-elastic attenuation
        parameters QP and QS are included in the data section.
        0 - no visco-elastic attenuation parameters included
        1 - visco-elastic attenuation parameters included

    ``az``
        Angle in degrees between North and the positive x-axis.

    ``lon0``
        Longitude of the origin of the data

    ``lat0``
        Latitude of the origin of the data

    ``mlen``
        The number of characters in the string ``proj_str``

    ``proj_str``
        Projection string which is read by the Proj4 library if SW4 was
        built with Proj4 support.

    ``nb``
        The number of blocks in the data section.

    Note
    ----
    See the `SW4 User Guide`_ for more details about these header
    parameters.

    .. _SW4 User Guide:
       https://geodynamics.org/cig/software/sw4/

    See Also
    --------
    :func:`.write_hdr`
    """

    (magic, precision, attenuation,
     az, lon0, lat0, mlen) = np.fromfile(f, RFILE_HEADER_DTYPE, 1)[0]
    proj_str_dtype = 'S' + str(mlen)
    proj_str = np.fromfile(f, proj_str_dtype, 1)[0]
    nb = np.fromfile(f, 'int32', 1)[0]

    return magic, precision, attenuation, az, lon0, lat0, mlen, proj_str, nb


def write_block_hdr(f, hh, hv, z0, nc, ni, nj, nk):
    """
    Write rfile block header

    Block headers are appended after the rfile header has been written.
    All block headers are written one after the other.

    Parameters
    ----------
    f : file
        Open file handle in ``'wa'`` mode.

    hh : numpy.float64
        Grid size in the horizontal directions (x and y) in meters.

    hv : numpy.float64
        Grid size in the vertical direction (z) in meters.

    z0 : numpy.float64
        The base z-level of the block. Not used for the first block
        which holds the elevation of the topography/bathymetry.

    nc : int
        The number of components: The first block holds the elevation
        of the topography/bathymetry, so ``nc=1``. The following blocks
        must have either 3 if only rho, vp, and vs are present
        (``attenuation=0``) or 5 if qp and qs are pressent
        (``attenuation=1``).

    ni : int
        Number of grid points in the i direction.

    nj : int
        Number of grid points in the j direction.

    nk : int
        Number of grid points in the k direction. Because the
        topography/bathymetry is only a function of the horizontal
        coordinates, the first block must have ``nk=1``.
    """

    f.write(np.float64(hh))
    f.write(np.float64(hv))
    f.write(np.float64(z0))
    f.write(np.int32(nc))
    f.write(np.int32(ni))
    f.write(np.int32(nj))
    f.write(np.int32(nk))


def read_block_hdr(f):
    """
    Read rfile block header.

    Parameters
    ----------
    f : file
        Open file handle in ``'rb'`` mode of the rfile to be read.

    Returns
    -------
    tuple
        A tuple containing rfile block header data (7 elements):
        hh, hv, z0, nc, ni, nj, nk


    .. rubric:: **Description of block header data returned**

    ``hh``
        Grid size in the horizontal directions (x and y) in meters.

    ``hv``
        Grid size in the vertical direction (z) in meters.

    ``z0``
        The base z-level of the block. Not used for the
        first block which holds the elevation of the topography/
        bathymetry.

    ``nc``
        The number of components:
        The first block holds the elevation of the topography/
        bathymetry, so ``nc=1``.
        The following blocks must have either 3 if only rho, vp, and vs
        are present (``attenuation=0``) or 5 if qp and qs are pressent
        (``attenuation=1``).

    ``ni``
        Number of grid points in the i direction.

    ``nj``
        Number of grid points in the j direction.

    ``nk``
        Number of grid points in the k direction.
        Because the topography/bathymetry is a function of the
        horizontal coordinates, the first block must have ``nk=1``.
    """

    return np.fromfile(f, BLOCK_HEADER_DTYPE, 1)[0]


def read(filename, block_number=None, verbose=False):
    """
    Docstring will be overwritten by ``Model.__init__.__doc__``
    """
    model = Model(filename, block_number, verbose)
    return model


class Model(object):
    """
    A class to hold rfile header and blocks.

    .. rubric:: **To read rfile into Model object**

    Parameters
    ----------
    filename : file
        Path to rfile

    block_number : int, str, None
        By default (``block_number=None``) only the rfile header and
        the block headers are read into a
        :class:`~pySW4.preprocess.rfileIO.Model` object. No block data
        sections are read.
        If ``block_number='all'``, all block data sections are read
        into memory.
        Therefor, you could read only one block by specifying
        ``block_number=?`` replacing ``?`` with the block number you
        are interested in.

    verbose : bool
        If ``True``, information about the read process is printed to
        ``stdout``.

    Returns
    -------
    :class:`~pySW4.preprocess.rfileIO.Model`
        A Model object holding rfile header and block headers and
        perhaps also block data sections, depending on the
        ``block_number`` value.

    Note
    ----
    * ``block_number='all'`` for large models may take a while and take
        up a lot of memory.
    * Block number 1 holds the elevation of the topography/bathymetry.
    """

    def __init__(self, filename, block_number=None, verbose=False):
        self.filename = filename
        self.blocks = []

        with open(filename, 'rb') as self.f:
            # read rfile header
            (self.magic,
             self.precision,
             self.attenuation,
             self.az,
             self.lon0,
             self.lat0,
             self.mlen,
             self.proj_str,
             self.nb) = read_hdr(self.f)

            if verbose:
                print(self)
                flush

            # read all block headers
            for b in range(self.nb):
                block = Block(self.f, b + 1)
                self.blocks += [block]

                if verbose:
                    print(block)
                    flush

            self.data_section_start = self.f.tell()
            if block_number is not None:
                self.read_block_data_section(block_number)

    def __str__(self):
        string = ('Model information:\n'
                  '-----------------\n'
                  '        Filename : {}\n'
                  '           lon 0 : {}\n'
                  '           lat 0 : {}\n'
                  '         Azimuth : {}\n'
                  '    Proj4 string : {}\n'
                  'Number of blocks : {}\n'.format(self.filename,
                                                   self.lon0,
                                                   self.lat0,
                                                   self.az,
                                                   self.proj_str,
                                                   self.nb))
        return string

    def read_block_data_section(self, block_number):
        """
        Read the data section of the specified block into the current
        :class:`pySW4.preprocess.rfileIO.Model` object to be stored in
        memory.

        Parameters
        ----------
        block_number : int or str
            The number of the block you are interested in. Block number
            1 holds the elevation of the topography/bathymetry. Blocks 2
            and on (``2<=block_number<=self.nb``) hold the material
            properties of the model. If ``block_number='all'``, all
            block data sections are read into memory.

        Note
        ----
        For large models this may take a while and take up a lot of
        memory.
        """

        # make sure the cursor is cued o the right position in the
        # re-opened file.
        if self.f.closed:
            self.f = open(self.filename, 'rb')
            self.f.seek(self.data_section_start)

        if block_number is 'all':
            # read data section block by block and
            # populate the proper block object
            for block in self.blocks:
                block._read_block_data_section(self.f, self.precision)
        elif isinstance(block_number, int):
            block = self.blocks[block_number - 1]
            for b in self.blocks:
                if b.number < block_number:
                    self.f.seek(self.precision
                                * b.ni
                                * b.nj
                                * b.nk
                                * b.nc, 1)
                    break
            block._read_block_data_section(self.f, self.precision)

    def get_properties_at_point(self, x, y, z, property='all'):
        """
        Get the material properties at a point.

        Parameters
        ----------
        x : float
            Distance in km along the x-axis of the model.

        y : float
            Distance in km along the y-axis of the model.

        z : float
            Distance in km along the z-axis of the model. (positive is
            down).

        property : str
            ``'all'`` returns all 5 material properties.
            Otherwise, only ``'rho'``, ``'vp'``, ``'vs'``, ``'qp'``, or
            ``'qs'`` are returned.

        Returns
        -------
        :class:`~numpy.ndarray` or float
            One or all of the material properties at point (x,y,z)
            (``rho``, ``vp``, ``vs``, ``qp``, ``qs``)
        """

        # find the right block
        for block in self.blocks[1:]:
            if block.z_extent[1] <= z <= block.z_extent[0]:
                break

        indices = (int(round(x * 1e3 / block.hh)),
                   int(round(y * 1e3 / block.hh)),
                   int(round((z * 1e3 - block.z0) / block.hv)))

        properties = block.data[indices]

        try:
            return properties[COMPONENTS[property]]
        except KeyError:
            return properties

    def get_z_profile(self, x, y, property='all'):
        """
        Get the material properties along a z-profile at location x,y.

        Parameters
        ----------
        x : float
            Distance in km along the x-axis of the model.

        y : float
            Distance in km along the y-axis of the model.

        property : str
            ``'all'`` returns all 5 material properties.
            Otherwise, only ``'rho'``, ``'vp'``, ``'vs'``, ``'qp'``, or
            ``'qs'`` are returned.

        Returns
        -------
        2 :class:`~numpy.ndarray`
            2 arrays:

            * z elevations, shape (n,)

            and

            * the corresponding material properties (one or all)
                (``rho``, ``vp``, ``vs``, ``qp``, ``qs``),
                shape (n,) or (n, 5)
        """

        properties = np.empty(5)
        for block in self.blocks[1:]:
            try:
                indices = (int(round(x * 1e3 / block.hh)),
                           int(round(y * 1e3 / block.hh)))

                properties = np.vstack((properties,
                                        block.data[indices]))
            except IndexError:
                msg = ('Block number {} has not been read so no data '
                       'is present. Replacing with -999. '
                       'Consider reading in the entire rfile first.')
                warn(msg.format(block.number))
                properties = np.vstack((properties,
                                        np.ones((block.z().shape[0],
                                                 5)) * -999.))
        try:
            return self.z(), properties[1:, COMPONENTS[property]]
        except KeyError:
            return self.z(), properties[1:]

    def get_cross_section(self, x1=None, x2=None, y1=None, y2=None):
        """
        Docstring will be overwritten by
        ``CrossSection.__init__.__doc__``
        """
        return CrossSection(self, x1, x2, y1, y2)

    def z(self):
        """
        Return a vector (:class:`~numpy.ndarray`) with z coordinates of
        the model based on the ``hv`` of each block. Coordinates are
        distance in km along the z-axis of the model which are elevation
        below sea level (positive is down).
        """

        zi = np.array([])
        for block in self.blocks[1:]:
            zi = np.hstack((zi, block.z()))
        return zi

    def get_topography(self):
        """
        Returns a 2d :class:`~numpy.ndarray` with elevation data
        retrieved from the topography/bathymetry block
        (the first block).

        Note
        ----
        The origin of the data is the bottom-left corner so if plotted
        with :func:`~matplotlib.pyplot.imshow` set the 'origin' keyword
        to 'lower' or simply use
        :meth:`~pySW4.preprocess.rfileIO.Model.plot_topography` for
        plotting.
        """

        return model.blocks[0].data

    def plot_topography(self, ax=None, **kwargs):
        """
        Plot the top surface (topography and bathymetry) of the model.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes. If ``ax=None`` function returns a
            :class:`~matplotlib.figure.Figure` instance,
            a :class:`~matplotlib.axes.Axes` instance,
            and a :class:`~matplotlib.colorbar.Colorbar` instance.
            Otherwise, only a :class:`~matplotlib.colorbar.Colorbar`
            instance is returned.


        .. rubric:: **Other useful keyword arguments are:**

        Keyword Arguments
        -----------------
        vmin : int or float
            Used to limit the scale of the data. By default the minimum
            and maximum of the data is used.

        vmax : int or float
            Used to limit the scale of the data. By default the minimum
            and maximum of the data is used.

        interpolation : str
            Acceptable values are 'none', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
            'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
            'bessel', 'mitchell', 'sinc', 'lanczos'.

        pad : int or float
            Horizontal space between the axes and the colorbar.

        size : str or float
            Width of the colorbar. If float, width is set to ``size``.
            If string colorbar width is calculated as a fraction of the
            axes width.

            For example:
                ``size='3%'`` will result in a colorbar who's width is
                    3% of the width of the axes.

            Similarly:
                ``size='0.3'`` will result in a colorbar who's width is
                    0.3 of the width of the axes.

        label : str
            Label used for the colorbar. By default the label is set by
            the property plotted.

        Note
        ----
        *For other keyword arguments* see:
        :func:`~matplotlib.pyplot.imshow`.
        """

        data = self.blocks[0].data
        extent = self.blocks[0].xyextent

        # parse kwargs
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = data.min()

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = data.max()

        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
        else:
            pad = 0.02
        if 'size' in kwargs:
            size = kwargs.pop('size')
        else:
            size = 0.025

        # setup the figure and the axes
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # plot the data
        im = ax.imshow(data, vmin=vmin, vmax=vmax,
                       extent=extent, **kwargs)

        ax.axis(extent)

        ax.set_xlabel('y distance from origin [km]')
        ax.set_ylabel('x distance from origin [km]')
        ax.set_title('Model topography/bathymetry')

        # force figure to update its layout
        plt.draw()

        ax_pos = ax.get_position()

        # setup the colorbar
        if vmin > data.min() and vmax < data.max():
            extend = 'both'
        elif vmin > data.min():
            extend = 'min'
        elif vmax < data.max():
            extend = 'max'
        else:
            extend = 'neither'

        if type(size) is str:
            size = float(size.strip('%'))
            if size >= 1:
                size /= 100
            size = ax_pos.width * size

        x0 = ax_pos.x1 + pad
        width = size
        y0 = ax_pos.y0
        height = ax_pos.height

        cax = fig.add_axes((x0, y0, width, height))
        if 'label' in kwargs:
            label = kwargs.pop('label')
        else:
            label = 'Elevation [m.a.s.l]'
        cb = plt.colorbar(im, cax=cax, extend=extend, label=label)
        cb.solids.set_edgecolor('face')

        try:
            return fig, ax, cb
        except UnboundLocalError:
            return cb


class Block(object):
    """
    A class to hold rfile block header and data section
    """

    def __init__(self, f, b):
        self.number = b
        (self.hh,
         self.hv,
         self.z0,
         self.nc,
         self.ni,
         self.nj,
         self.nk) = read_block_hdr(f)
        self.data = np.array([])

        self.x_extent = (0, (self.ni - 1) * self.hh * 1e-3)
        self.y_extent = (0, (self.nj - 1) * self.hh * 1e-3)
        self.z_extent = ((self.z0 + (self.nk - 1) * self.hv) * 1e-3,
                         self.z0 * 1e-3)

        self.xyextent = self.y_extent + self.x_extent

        self.xzextent = self.x_extent + self.z_extent

        self.yzextent = self.y_extent + self.z_extent

    def __str__(self):
        string = ('Block information:\n'
                  '-----------------\n'
                  '          Number : {}\n'
                  ' Horizontal h, m : {}\n'
                  '   Vertical h, m : {}\n'
                  '          z 0, m : {}\n'
                  '              ni : {}\n'
                  '              nj : {}\n'
                  '              nk : {}\n'
                  '              nc : {}\n'.format(self.number,
                                                   self.hh,
                                                   self.hv,
                                                   self.z0,
                                                   self.ni,
                                                   self.nj,
                                                   self.nk,
                                                   self.nc))
        return string

    def _read_block_data_section(self, f, precision):
        """
        Private method. Reads the data section of cued file handle.
        Should not be used by the user. See
        :meth:`~.Model.read_block_data_section` method in the
        :class:`~pySW4.preprocess.rfileIO.Model` class.
        """
        data = np.fromfile(f, DATA_PRECISION[precision],
                           self.ni * self.nj * self.nk * self.nc)

        if self.nc == 1:  # topo is independant of k
            self.data = data.reshape(self.ni, self.nj)
        else:
            # C-order reshape
            self.data = data.reshape(self.ni, self.nj, self.nk, self.nc)

    def vp(self):
        """
        Return values of vp for block
        """
        return self.data[:, :, :, 1]

    def vs(self):
        """
        Return values of vs for block
        """
        return self.data[:, :, :, 2]

    def rho(self):
        """
        Return values of density for block
        """
        return self.data[:, :, :, 0]

    def qp(self):
        """
        Return values of Qp for block
        """
        return self.data[:, :, :, 3]

    def qs(self):
        """
        Return values of Qs for block
        """
        return self.data[:, :, :, 4]

    def x(self):
        """
        Return a vector with x coordinates of the block based on ``hh``.
        Coordinates are distance in km.
        """

        hh = self.hh * 1e-3
        return np.arange(self.x_extent[0], self.x_extent[1] + hh, hh)

    def y(self):
        """
        Return a vector with y coordinates of the block based on ``hh``.
        Coordinates are distance in km.
        """

        hh = self.hh * 1e-3
        return np.arange(self.y_extent[0], self.y_extent[1] + hh, hh)

    def z(self):
        """
        Return a vector with z coordinates of the block based on ``hv``.
        Coordinates are elevation below sea level in km
        (positive is down).
        """

        hv = self.hv * 1e-3
        return np.arange(self.z_extent[1], self.z_extent[0] + hv, hv)


class CrossSection(object):
    """
    Class to generate and hold a cross-section of the material
    properties along a line in an existing model
    (:class:`~pySW4.preprocess.rfileIO.Model`).

    Parameters
    ----------
    model : :class:`pySW4.preprocess.rfileIO.Model`
        A populated Model object from which to extract the data

    x1 : float
        Distance in km along the x-axis of start-point.

    x2 : float
        Distance in km along the x-axis of end-point.

    y1 : float
        Distance in km along the y-axis of start-point.

    y2 : float
        Distance in km along the y-axis of end-point.

    returns
    -------
    :class:`~pySW4.preprocess.rfileIO.CrossSection`
        A ``CrossSection`` object with data and a plotting method.
    """

    def __init__(self, model, x1=None, x2=None, y1=None, y2=None):
        hh = model.blocks[1].hh * 1e-3
        # make sure all points are properly defined
        if x1 is x2 is y1 is y2 is None:
            msg = 'Must provide at least one of (x1, x2, y1, y2)'
            raise ValueError(msg)
        else:
            if x1 is None and x2 is not None:
                x1 = x2
            elif x1 is not None and x2 is None:
                x2 = x1
            elif x1 is None and x2 is None:
                x1, x2 = model.blocks[1].x_extent

            if y1 is None and y2 is not None:
                y1 = y2
            elif y1 is not None and y2 is None:
                y2 = y1
            elif y1 is None and y2 is None:
                y1, y2 = model.blocks[1].y_extent

        if x1 == x2 and y1 == y2:
            msg = 'No line is formed between (x1={},y1={}) and (x2={},y2={}).'
            raise ValueError(msg.format(x1, y1, x2, y2))

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        # find all pixels along the cross-section line in each block
        y, x = line_func(y1, y2, x1, x2, hh)
        self.z = model.z()

        self.cs_coordinates = np.sqrt(y**2 + x**2)
        self.h_extent = (self.cs_coordinates[0], self.cs_coordinates[-1])

        # initialize the array for the data and populate it
        properties = np.empty((self.z.size, self.cs_coordinates.size, 5))
        for col, (xi, yj) in enumerate(zip(x, y)):
            properties[:, col] = model.get_z_profile(xi, yj)[1]

        self.max = [col.max() for col in properties.T]
        self.min = [col.min() for col in properties.T]
        self.std = [col.std() for col in properties.T]

        # split the properties array into block with different extents
        indices = np.cumsum([block.nk for block in model.blocks[1:-1]])
        self.extents = [self.h_extent
                        + block.z_extent for block in model.blocks[1:]]

        self.data = np.vsplit(properties, indices)

    def plot(self, property='vp', ax=None, draw_separator=False, **kwargs):
        """
        Plot cross-section.

        Parameters
        ----------
        property : str
            Which property to plot:
            ``'vp'``, ``'vs'``, ``'qp'``, or ``'qs'``.

        ax : :class:`~matplotlib.axes.Axes`
            Use existing axes. If ``ax=None`` function returns:
            a :class:`~matplotlib.figure.Figure` instance,
            a :class:`~matplotlib.axes.Axes` instance,
            and a :class:`~matplotlib.colorbar.Colorbar` instance.
            Otherwise, only a :class:`~matplotlib.colorbar.Colorbar`
            instance is returned.

        draw_separator : bool
            If set to ``True``, a separator is plotted between
            blocks.

        .. rubric:: **Other useful keyword arguments are:**

        Keyword Arguments
        -----------------
        vmin : int or float
            Used to limit the scale of the data. By default the minimum
            and maximum of the data is used.

        vmax : int or float
            Used to limit the scale of the data. By default the minimum
            and maximum of the data is used.

        aspect : str or int or float
            Changes the axes aspect ratio.

        interpolation : str
            Acceptable values are 'none', 'nearest', 'bilinear',
            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
            'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
            'bessel', 'mitchell', 'sinc', 'lanczos'.

        pad : int or float
            Horizontal space between the axes and the colorbar.

        size : str or float
            Width of the colorbar. If float, width is set to ``size``.
            If string colorbar width is calculated as a fraction of the
            axes width.

            For example:
                ``size='3%'`` will result in a colorbar who's width is
                    3% of the width of the axes.

            Similarly:
                ``size='0.3'`` will result in a colorbar who's width is
                    0.3 of the width of the axes.

        label : str
            Label used for the colorbar. By default the label is set by
            the property plotted.

        Note
        ----
        *For other keyword arguments* see:
        :func:`~matplotlib.pyplot.imshow`.
        """

        # parse kwargs
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = self.min[COMPONENTS[property]]

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = self.max[COMPONENTS[property]]

        if 'aspect' in kwargs:
            aspect = kwargs.pop('aspect')
        else:
            aspect = 1

        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
        else:
            pad = 0.02
        if 'size' in kwargs:
            size = kwargs.pop('size')
        else:
            size = 0.025

        # setup the figure and the axes
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # plot each block
        for i, data in enumerate(self.data):
            data[data == -999] = np.nan
            im = ax.imshow(data[:, :, COMPONENTS[property]],
                           vmin=vmin, vmax=vmax,
                           extent=self.extents[i], **kwargs)

        if draw_separator:
            extents = np.array(self.extents)
            ax.hlines(extents[:, 2], xmin=extents[:, 0], xmax=extents[:, 1])

        ax.axis(self.h_extent + (self.z[-1], self.z[0]))
        ax.set_aspect(aspect)

        ax.set_xlabel('Distance from origin [km]')
        ax.set_ylabel('Depth from sea-level [km]')
        ax.set_title('Cross-section along line from\n'
                     'P1({},{}) to P2({},{})'.format(self.x1, self.y1,
                                                     self.x2, self.y2))

        # force figure to update its layout
        plt.draw()

        ax_pos = ax.get_position()

        # setup the colorbar
        if (vmin > self.min[COMPONENTS[property]] and
                vmax < self.max[COMPONENTS[property]]):
            extend = 'both'
        elif vmin > self.min[COMPONENTS[property]]:
            extend = 'min'
        elif vmax < self.max[COMPONENTS[property]]:
            extend = 'max'
        else:
            extend = 'neither'

        if type(size) is str:
            size = float(size.strip('%'))
            if size >= 1:
                size /= 100
            size = ax_pos.width * size

        x0 = ax_pos.x1 + pad
        width = size
        y0 = ax_pos.y0
        height = ax_pos.height

        cax = fig.add_axes((x0, y0, width, height))
        if 'label' in kwargs:
            label = kwargs.pop('label')
        else:
            label = ('{} [{}]'
                     .format(TITLES_AND_LABELS[
                             COMPONENTS[property]]['symbol'],
                             TITLES_AND_LABELS[
                             COMPONENTS[property]]['unit']))
        cb = plt.colorbar(im, cax=cax, extend=extend, label=label)
        cb.solids.set_edgecolor('face')

        try:
            return fig, ax, cb
        except UnboundLocalError:
            return cb


# The following function is copied over from ``pySW4.utils.utils.py``
# to help make this IO library independant of the rest of the package.
def line_func(h1, h2, v1, v2, spacing=1):
    """
    Return all coordinates along a straight line from point
    (``h1``,``v1`) to point (``h2``,``v2``) with ``spacing``.

    Parameters
    ----------
    h1 : int or float
        Horizontal coordinate of the start-point.

    h2 : int or float
        Horizontal coordinate of the end-point.

    v1 : int or float
        Vertical coordinate of the start-point.

    v2 : int or float
        Vertical coordinate of the end-point.

    spacing : int or float
        Spacing along the straight line between coordinates.
        Controls the data-type returned: If ``spacing`` type is ``int``,
        the values in the returned arrays are ``int``. If ``spacing``
        type is float, the values in the returned arrays are ``float``.

    Returns
    -------
    2 class:`~numpy.ndarray`
        2 arrays with horizontal coordinates and vertical coordinates.
    """

    v_diff = v2 - v1
    h_diff = h2 - h1

    if h_diff == 0:
        n = int(v_diff / spacing)
        v = np.linspace(v1, v2, n + 1)
        h = h1 * np.ones_like(v)
    elif v_diff == 0:
        n = int(h_diff / spacing)
        h = np.linspace(h1, h2, n + 1)
        v = v1 * np.ones_like(h)
    else:
        slope = float(v_diff) / (h_diff)
        if slope > 1:
            dh = spacing / slope
        else:
            dh = spacing * slope

        l = (h_diff**2 + v_diff**2)**0.5
        n = l / dh

        b = v1 - slope * h1
        h = np.linspace(h1, h2, n + 1)
        v = (slope * h + b)

    if type(spacing) is int:
        h = h.astype(int)
        v = v.astype(int)

    return h, v


# Set docstrings
read.__doc__ = Model.__doc__
Model.get_cross_section.__func__.__doc__ = CrossSection.__doc__
