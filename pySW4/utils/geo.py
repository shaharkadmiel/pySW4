"""
Python module for handling DEM/GDEM/DTM and other raster data readable
with gdal.

.. module:: geo

:author:
    Shahar Shani-Kadmiel
    s.shanikadmiel@tudelft.nl

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import absolute_import, print_function, division

from scipy import ndimage
import numpy as np
import sys
import os
try:  # on python 3
    from zipfile import ZipFile, BadZipFile
except ImportError:  # on python 2
    from zipfile import ZipFile, BadZipfile 
import tarfile
from tarfile import ReadError
from fnmatch import fnmatch
import shutil
import copy
from warnings import warn

from ..plotting import calc_intensity

try:
    from osgeo import gdal, osr, gdal_array
except ImportError:
    warn('gdal not found, you will not be able to use the geo tools '
         ' in `pySW4.utils.geo` unless you install gdal.')

GDAL_INTERPOLATION = {'nearest' : gdal.GRA_NearestNeighbour,
                      'bilinear': gdal.GRA_Bilinear,
                      'cubic'   : gdal.GRA_Cubic,
                      'lanczos' : gdal.GRA_Lanczos}


def gdalopen(filename, substring='*_dem.tif'):
    """
    Wrapper function around :func:`gdal.Open`.

    :func:`gdal.Open` returns ``None`` if no file is read. This wrapper
    function will try to read the `filename` directly but if ``None``
    is returned, several other options, including compressed file
    formats ('zip', 'gzip', 'tar') and url retrieval are attempted.
    """

    ds = gdal.Open(filename)
    if not ds:
        # try to open compressed file container and provide more info
        # in case read is unsuccessful
        try:
            f_ = ZipFile(filename)
            contents = f_.namelist()
            prefix_ = 'zip'

        except BadZipFile:
            try:
                f_ = tarfile.open(filename)
                contents = f_.getnames()
                prefix_ = 'tar'
            except ReadError:
                # gzip:
                prefix_ = 'gzip'
                filename_ = '/vsi{}/'.format(prefix_) + filename

                ds = gdal.Open(filename_)
                if ds:
                    return ds

                # url:
                prefix_ = 'curl'
                filename_ = '/vsi{}/'.format(prefix_) + filename

                ds = gdal.Open(filename_)
                if ds:
                    return ds

        for item in contents:
            if fnmatch(item, substring) or substring in item:
                filename_ = ('/vsi{}/'.format(prefix_) +
                             os.path.join(filename, item))
                ds = gdal.Open(filename_)
                if ds:
                    return ds

        msg = (
            "No file with matching substring '{}' was found in {}, "
            'which contains:\n'
            '   {}\n'
            "try:\n{}"
        ).format(
            substring, filename, contents,
            '\n'.join(['/vsi{}/'.format(prefix_) + os.path.join(
                filename, item) for item in contents]))

        raise OSError(msg)

    return ds


class GeoTIFF():
    """
    Class for handling GeoTIFF files.

    In the GeoTIFF file format, it is assumed that the origin of the
    grid is at the top-left (north-west) corner. Hence, ``dx`` is
    positive in the ``x`` or ``lon`` direction and ``dy`` is negative
    in the ``y`` or ``lat`` direction.

    .. note:: This class should be populated by
              :func:`~pySW4.utils.geo.read_GeoTIFF` or
              :func:`~pySW4.utils.geo.get_dem`.
    """

    def __init__(self):
        self.path = './'
        self.name = ''
        self.nodata = np.nan
        self.w = 0.
        self.n = 0.
        self.proj4 = b'+proj=longlat +datum=WGS84 +no_defs'
        self.dx = 0.
        self.dy = -0.

        self.z = np.empty((1, 1), dtype=np.int16)

    @property
    def elev(self):
        return self.z

    @property
    def nx(self):
        return self.z.shape[1]

    @property
    def ny(self):
        return self.z.shape[0]

    @property
    def e(self):
        return self.w + (self.nx) * self.dx

    @property
    def s(self):
        return self.n + (self.ny) * self.dy

    @property
    def extent(self):
        return self.w, self.e, self.s, self.n

    @property
    def dtype(self):
        return self.z.dtype

    def _read(self, filename, rasterBand, substring='*_dem.tif',
              verbose=False):
        """
        Private method for reading a GeoTIFF file.
        """
        try:
            self.path, self.name = os.path.split(filename)

            if verbose:
                print('Reading GeoTIFF file: %s' % filename)
                sys.stdout.flush()

            src_ds = gdalopen(filename, substring)
            if not src_ds:
                filename = '/vsizip/' + filename
                print('trying zipfile', filename)
                src_ds = gdal.Open(filename)
        except AttributeError:
            if verbose:
                print('Updating GeoTIFF file...')
                sys.stdout.flush()
            src_ds = filename

        self._src_ds = src_ds
        band = src_ds.GetRasterBand(rasterBand)
        # self.dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)
        self.z = band.ReadAsArray()

        self.nodata = band.GetNoDataValue()
        self.geotransform = src_ds.GetGeoTransform()
        self.w = self.geotransform[0]
        self.n = self.geotransform[3]
        self.dx = self.geotransform[1]
        self.dy = self.geotransform[5]

        self.proj4 = (osr.SpatialReference(
            self._src_ds.GetProjection()).ExportToProj4() or
            self.proj4)

    def __str__(self):
        string = '\nGeoTIFF info:\n'
        if self.name:
            string += 'name: {}\n'.format(self.name)
        string += ('west: {}\n'
                   'east: {}\n'
                   'south: {}\n'
                   'north: {}\n'
                   'x pixel size: {}\n'
                   'y pixel size: {}\n'
                   '# of x pixels: {}\n'
                   '# of y pixels: {}\n'
                   'no data value: {}\n'
                   'data type: {}\n'
                   'proj4: {}\n'.format(self.w,
                                        self.e,
                                        self.s,
                                        self.n,
                                        self.dx,
                                        self.dy,
                                        self.nx,
                                        self.ny,
                                        self.nodata,
                                        self.dtype,
                                        self.proj4))
        return string

    @property
    def x(self):
        """
        Returns `x` coordinates vector of the data. This usually
        corresponds to Longitude.
        """
        return np.linspace(self.w, self.e, self.nx)

    @property
    def y(self):
        """
        Returns `y` coordinates vector of the data. This usually
        corresponds to Latitude.
        """
        return np.linspace(self.n, self.s, self.ny)

    @property
    def xy2d(self):
        """
        Returns 2 2d :class:`~numpy.ndarray` s with `x` and `y`
        coordinates of the data. This is the result of
        :func:`~numpy.meshgrid` of the `x` and `y` vectors of the
        data.
        """
        return np.meshgrid(self.x, self.y)

    def resample(self, by=None, to=None, order=0):
        """
        Method to resample the data either `by` a factor or `to`
        the specified spacing.

        This method uses :func:`~scipy.ndimage.zoom`.

        .. warning:: **Watch Out!** This operation is performed in place
                     on the actual data. The raw data will no longer be
                     accessible afterwards. To keep the original data,
                     use the :meth:`~pySW4.utils.geo.GeoTIFF.copy`
                     method to create a copy of the current object.

        Parameters
        ----------
        by : float

            - If ``by = 1``, nothing happens.
            - If ``by < 1``, the grid is sub-sampled by the factor.
            - if ``by > 1``, the grid is super-sampled by the factor.

        to : float
            The specified spacing to which the grid is resampled to.

        order : int
            The order of the spline interpolation, default is 0.
            The order has to be in the range 0-5.

            - 0 - nearest (fastest)
            - 1 - bi-linear
            - 3 - cubic (slower)
            - ...

            **Note** that this may result in slightly different
            spacing than desired and more importantly, may cause a
            **discrepancy** between `x` and `y` spacing.

        """

        if by and to:
            msg = 'Only `by` or `to` should be specified, not both.'
            raise ValueError(msg)

        if to:
            by = self.dx / to

        self.z = ndimage.zoom(self.z, by, order=order)

        # update class data
        self.dx /= by
        self.dy /= by

    def reproject(self, epsg=None, proj4=None, match=None, spacing=None,
                  interpolation='nearest', error_threshold=0.125,
                  target_filename=None, verbose=False):
        """
        Reproject the data from the current projection to the specified
        target projection `epsg` or `proj4` or to `match` an
        existing GeoTIFF file. Optionally, the grid spacing can be
        changed as well.

        To keep the same projection and only resample the data leave
        `epsg`, `proj4`, and `match` to be ``None`` and specify
        `spacing`.

        .. warning:: **Watch Out!** This operation is performed in place
                     on the actual data. The raw data will no longer be
                     accessible afterwards. To keep the original data,
                     use the :meth:`~pySW4.utils.geo.GeoTIFF.copy`
                     method to create a copy of the current object.

        Parameters
        ----------
        epsg : int
            The target projection EPSG code. See the
            `Geodetic Parameter Dataset Registry
            <http://www.epsg-registry.org/>`_ for more information.

        proj4 : str
            The target Proj4 string. If the EPSG code is unknown or a
            custom projection is required, a Proj4 string can be passed.
            See the `Proj4 <https://trac.osgeo.org/proj/wiki/GenParms>`_
            documentation for a list of general Proj4 parameters.

        match : str or :class:`~pySW4.utils.geo.GeoTIFF` instance
            Path (relative or absolute) to an existing GeoTIFF file or
            :class:`~pySW4.utils.geo.GeoTIFF` instance (already in
            memory) to match size and projection of. Current data is
            resampled to match the shape and number of pixels of the
            existing GeoTIFF file or object.
            It is assumed that both GeoTIFF objects cover the same
            extent.

        spacing : int or float
            Target spacing in the units of ther target projection.

        interpolation : {'nearest', 'bilinear', 'cubic', 'lanczos'}
            Resampling algorithm:

            - 'nearest' : Nearest neighbour
            - 'bilinear' : Bilinear (2x2 kernel)
            - 'cubic' : Cubic Convolution Approximation (4x4 kernel)
            - 'lanczos' : Lanczos windowed sinc interpolation (6x6 kernel)

        error_threshold : float
            Error threshold for transformation approximation in pixel
            units. (default is 0.125 for compatibility with  gdalwarp)

        target_filename : str
            If a target filename is given then the reprojected data is
            saved as target_filename and read into memory replacing
            the current data. This is faster than reprojecting and then
            saving. Otherwise the
            :meth:`~pySW4.utils.geo.GeoTIFF.write_GeoTIFF` method can be
            used to save the data at a later point if further
            manipulations are needed.

        Notes
        -----
        Examples of some EPSG codes and their equivalent Proj4 strings
        are::

            4326   -> '+proj=longlat +datum=WGS84 +no_defs'

            32636  -> '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs'

            102009 -> '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40
                       +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m
                       +no_defs'

        and so on ... See the `Geodetic Parameter Dataset Registry
        <http://www.epsg-registry.org/>`_ for more information.
        """
        if epsg and proj4 and match:
            msg = 'Only `epsg`, `proj4`, or `match` should be specified.'
            raise ValueError(msg)
        elif not epsg and not proj4 and not match and not spacing:
            msg = '`epsg`, `proj4`, `spacing` or `match` MUST be specified.'
            raise ValueError(msg)

        # Set up the two Spatial Reference systems
        srcSRS = osr.SpatialReference()
        srcSRS.ImportFromProj4(self.proj4)

        dstSRS = osr.SpatialReference()

        interpolation = GDAL_INTERPOLATION[interpolation]

        if epsg or proj4 or spacing:
            try:
                dstSRS.ImportFromEPSG(epsg)
            except TypeError:
                # Default to the same projection as the source
                if not proj4:
                    proj4 = self.proj4
                dstSRS.ImportFromProj4(proj4)

            # Default to same spacing
            spacing = float(spacing) or self.dx

            # Work out the boundaries of the target dataset
            tx = osr.CoordinateTransformation(srcSRS, dstSRS)

            (ulx, uly, ulz) = tx.TransformPoint(self.w, self.n)
            (lrx, lry, lrz) = tx.TransformPoint(self.w + self.dx * self.nx,
                                                self.n + self.dy * self.ny)

            gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype)

            if not target_filename:
                # Create an in-memory raster
                drv = gdal.GetDriverByName('MEM')
                dst_ds = drv.Create('',
                                    int((lrx - ulx) / spacing),
                                    int((uly - lry) / spacing), 1,
                                    gdal_dtype)
            else:
                # Create an raster file
                drv = gdal.GetDriverByName('GTiff')
                dst_ds = drv.Create(target_filename,
                                    int((lrx - ulx) / spacing),
                                    int((uly - lry) / spacing), 1,
                                    gdal_dtype)

            # Calculate the new geotransform
            dstGT = (ulx, spacing, 0, uly, 0, -spacing)

            # Set the geotransform
            dst_ds.SetGeoTransform(dstGT)
            dst_ds.SetProjection(dstSRS.ExportToWkt())

            # keep nodata value
            band = dst_ds.GetRasterBand(1)
            band.Fill(self.nodata)
            band.SetNoDataValue(self.nodata)
            band.FlushCache()

            # Perform the projection/resampling
            src_proj = self._src_ds.GetProjection()
            res = gdal.ReprojectImage(self._src_ds, dst_ds,
                                      src_proj, dstSRS.ExportToWkt(),
                                      interpolation,
                                      error_threshold)

            assert res == 0
            self._read(dst_ds, 1, verbose)

    def set_new_extent(self, w, e, s, n, fill_value=None, mask=False):
        """
        Change the extent of a GeoTIFF file, trimming the boundaries or
        padding them as needed.

        .. warning:: **Watch Out!** This operation is performed in place
            on the actual data. The raw data will no longer be
            accessible afterwards. To keep the original data, use the
            :meth:`~pySW4.utils.geo.GeoTIFF.copy` method to create a
            copy of the current object.

        Parameters
        ----------
        w, e, s, n : float
            The west-, east-, south-, and north-most coordinate to keep
            (may also be the xmin, xmax, ymin, ymax coordinate).

        fill_value : float or None
            Value to pad the data with in case extent is expanded beyond
            the data. By default this is set to the ``nodata`` value.

        mask : bool
            If set to ``True``, data is masked using the `fill_value`.
        """
        assert w < e, '`w` must be less than `e`'
        assert s < n, '`s` must be less than `n`'

        npw = int((w - self.w) / self.dx)
        npe = int((e - self.w) / self.dx)

        nps = int((s - self.n) / self.dy)
        npn = int((n - self.n) / self.dy)

        # trimming all dimensions
        if 0 <= npw < npe <= self.nx and 0 <= npn < nps <= self.ny:
            self.z = self.z[npn: nps, npw: npe]

        # expanding or trimming each dimension independently
        else:
            fill_value = self.nodata if fill_value is None else fill_value

            # first handle east and south boundaries
            if npe > self.nx:
                self.z = np.pad(self.z, ((0, 0), (0, npe - self.nx)),
                                'constant', constant_values=fill_value)
            else:
                self.z = self.z[:, :npe]
            if nps > self.ny:
                self.z = np.pad(self.z, ((0, nps - self.ny), (0, 0)),
                                'constant', constant_values=fill_value)
            else:
                self.z = self.z[:nps, :]

            # now handle west and north boundaries
            if npw < 0:
                self.z = np.pad(self.z, ((0, 0), (-npw, 0)),
                                'constant', constant_values=fill_value)
            else:
                self.z = self.z[:, npw:]
            if npn < 0:
                self.z = np.pad(self.z, ((-npn, 0), (0, 0)),
                                'constant', constant_values=fill_value)
            else:
                self.z = self.z[npn:, :]

            if mask:
                self.z = np.ma.masked_equal(self.z, fill_value)

        # update class data
        self.w = self.w + npw * self.dx
        self.n = self.n + npn * self.dy

    def keep(self, w, e, s, n):
        """
        Deprecated! Will dissapear in future vertions. Please use
        :meth:`.set_new_extent` method.
        """
        warn(('Deprecation Warning: Deprecated! Will dissapear in future'
              'vertions. Please use :meth:`.set_new_extent` method.'))
        self.set_new_extent(w, e, s, n)

    def elevation_profile(self, xs, ys):
        """
        Get the elevation values ar points in ``xs`` and ``ys``.
        """
        rows = []
        columns = []
        for x, y in zip(xs, ys):
            row = (x - self.w) / self.dx
            column = (self.n - y) / self.dy

            if row > self.z.shape[1] or column > self.z.shape[0]:
                continue
            # Add the point to our return array
            rows += [int(row)]
            columns += [int(column)]
        return self.z[columns, rows]

    def get_intensity(self, azimuth=315., altitude=45.,
                      scale=None, smooth=None, normalize=False):
        """
        This is a method to create an intensity array that can be used as
        illumination to create a shaded relief map or draping data over.
        It is assumed that the grid origin is at the upper-left corner.
        If that is not the case, add 90 to ``azimuth``.

        Parameters
        ----------
        relief : a 2d :class:`~numpy.ndarray`
            Topography or other data to calculate intensity from.

        azimuth : float
            Direction of light source, degrees from north.

        altitude : float
            Height of light source, degrees above the horizon.

        scale : float
            Scaling value of the data.

        smooth : float
            Number of cells to average before intensity calculation.

        normalize : bool
            By default the intensity is clipped to the [0,1] range. If set
            to ``True``, intensity is normalized to [0,1].

        Returns
        -------
        intensity : :class:`~numpy.ndarray`
            a 2d array with illumination data clipped in the [0,1] range.
            Optionally, can be normalized (instead of clipped) to [0,1].
            Same size as ``relief``.
        """
        intensity = calc_intensity(self.z, azimuth, altitude,
                                   scale, smooth, normalize)
        return intensity

    def write_topo_file(self, filename):
        """
        Write an ASCII Lon, Lat, z file for SW4 topography.
        """

        header = '%d %d' % (self.nx, self.ny)
        lon, lat = self.xy2d()
        np.savetxt(filename, np.column_stack((lon.ravel(),
                                              lat.ravel(),
                                              self.z.ravel())),
                   fmt='%f', header=header, comments='')

    def set_nodata(self, value):
        """
        Replace the current ``nodata`` value with the new `value`.
        """

        try:
            if np.isnan(self.z).any():
                self.z[np.isnan(self.z)] = value
            else:
                self.z[self.z == self.nodata] = value

            self.nodata = value
        except ValueError:
            self.z = self.z.astype(np.float32)
            self.dtype = np.float32
            self.set_nodata(value)

    def write_GeoTIFF(self, filename, nodata=None):
        """
        Write a GeoTIFF file.

        Parameters
        ----------
        filename : str
            Path (relative or absolute) to the saves file.

        nodata : int or float or NaN
            Value to register as the no-data value in the GeoTIFF file
            header.
        """

        if not nodata:
            nodata = self.nodata
            if not nodata:
                self.nodata = np.nan
                print('Setting nodata value to NaN')
        elif nodata != self.nodata:
            self.set_nodata(nodata)

        save_GeoTIFF(filename, self.z, self.w, self.n,
                     self.dx, self.dy, proj4=self.proj4,
                     dtype=self.dtype, nodata=self.nodata,
                     rasterBand=1)

    def copy(self):
        """
        Return a deepcopy of the GeoTIFF object.
        """
        new = GeoTIFF()
        new.__dict__ = self.__dict__.copy()
        # new.path = copy.deepcopy(self.path)
        # new.name = copy.deepcopy(self.name)
        # new.nodata = copy.deepcopy(self.nodata)
        # new.w = copy.deepcopy(self.w)
        # new.n = copy.deepcopy(self.n)
        # new.proj4 = copy.deepcopy(self.proj4)
        # new.dx = copy.deepcopy(self.dx)
        # new.dy = copy.deepcopy(self.dy)
        # new.z = copy.deepcopy(self.z)
        # try:
        #     new._src_ds = gdal.Open(os.path.join(self.path, self.name))
        # except AttributeError:
        #     new._src_ds = None
        return new


def read_GeoTIFF(filename, rasterBand=1, substring='*_dem.tif',
                 verbose=False):
    """
    Read a single GeoTIFF file.

    Parameters
    ----------
    filename : str
        Path (relative or absolute) to a GeoTIFF file.

        .. note::

            It is possible to read a GeoTIFF file directly from within
            a compressed archive without extracting it by using the
            ``/vsiPREFIX/``.

            For instance, to read 'my.tif' from a subdirectory within
            'my.zip' use: ``/vsizip/my.zip/subdirectory/my.tif`` if path
            to 'my.zip' is relative or
            ``/vsizip//full_path_to/my.zip/subdirectory/my.tif`` if
            using the absolute path. Note the extra ``/`` after
            ``/vsizip/``.

            See http://www.gdal.org/gdal_virtual_file_systems.html for
            more details.

    rasterBand : int
        The band number to read.

    substring : str
        Compressed file formats may archive more than one file. The
        filename needs to be explicit as described in the `note` above.
        However, if the filename is not explicit, this function will
        try to find the correct file based on substring or regular
        expression matching.

        .. note:: Being explicit is faster (but not by much)

    Returns
    -------
    :class:`~pySW4.utils.geo.GeoTIFF`
        A populated (or empty) :class:`~pySW4.utils.geo.GeoTIFF`
        instance with the elevation data.
    """

    tif = GeoTIFF()
    tif._read(filename, rasterBand, substring, verbose)

    return tif


def save_GeoTIFF(filename, data, tlx, tly, dx, dy,
                 epsg=None, proj4=None,
                 dtype=np.int16, nodata=np.nan, rasterBand=1):
    """
    Save data at a known projection as a GeoTIFF file.

    Parameters
    ----------
    filename : str
        Path (relative or absolute) of the output GeoTIFF file.

    data : :class:`~numpy.ndarray`
        A 2d array of data to write to file.

    tlx : float
        Top-left-x (usually West-) coordinate of data.

    tly : flaot
        Top-left-y (usually North-) coordinate of data.

    dx : float
        Pixel size of data in the x direction, positive in the East
        direction.

    dy : float
        Pixel size of data in the y direction, positive in the North
        direction.

    epsg : int
        The target projection EPSG code. See the
        `Geodetic Parameter Dataset Registry
        <http://www.epsg-registry.org/>`_ for more information.

    proj4 : str
        The target Proj4 string. If the EPSG code is unknown or a
        custom projection is required, a Proj4 string can be passed.
        See the `Proj4 <https://trac.osgeo.org/proj/wiki/GenParms>`_
        documentation for a list of general Proj4 parameters.

    dtype : :class:`~numpy.dtype`
        One of following dtypes should be used:

        - ``numpy.int16`` (default)
        - ``numpy.int32``
        - ``numpy.float32``
        - ``numpy.float64``

    nodata : int or float or NaN
        Value to register as the no-data value in the GeoTIFF file
        header. By default this is set to ``numpy.nan``. Other common
        options are -9999, -12345 or any other value that suits your
        purpose.

    rasterband : int
        The band number to write. (default is 1)

    Notes
    -----
    Examples of some EPSG codes and their equivalent Proj4 strings
    are::

        4326   -> '+proj=longlat +datum=WGS84 +no_defs'

        32636  -> '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs'

        102009 -> '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40
                   +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m
                   +no_defs'

    and so on ... See the `Geodetic Parameter Dataset Registry
    <http://www.epsg-registry.org/>`_ for more information.
    """

    if epsg and proj4:
        msg = 'Only `epsg` or `proj4` should be specified, not both.'
        raise ValueError(msg)
    elif not epsg and not proj4:
        msg = '`epsg` or `proj4` MUST be specified.'
        raise ValueError(msg)

    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    if not gdal_dtype:
        msg = ('Illegal DataType {}. ``dtype`` can be one of '
               '``numpy.int16``, `numpy.int32`, `numpy.float32`, or '
               '`numpy.float64`.')
        raise TypeError(msg.format(dtype))

    dst_ds = gdal.GetDriverByName('GTiff').Create(filename,
                                                  data.shape[1],
                                                  data.shape[0],
                                                  1,
                                                  gdal_dtype)
    dst_ds.SetGeoTransform((tlx, dx, 0, tly, 0, dy))

    dstSRS = osr.SpatialReference()

    if epsg:
        dstSRS.ImportFromEPSG(epsg)
    elif proj4:
        dstSRS.ImportFromProj4(proj4)

    dst_ds.SetProjection(dstSRS.ExportToWkt())

    dstband = dst_ds.GetRasterBand(rasterBand)
    dstband.WriteArray(data.astype(dtype))
    dstband.SetNoDataValue(nodata)

    dst_ds = None


tilename_template = 'ASTGTM2_{}{:02d}{}{:03d}'

def _get_tiles(path, lonmin, lonmax, latmin, latmax,
               tilename=tilename_template, verbose=False):
    """
    This is a helper function for :func:`~pySW4.utils.geo.get_dem`
    function.

    It generates a list of tile filenames that are needed for the
    stitching process.

    .. note::
        This function should not be used directly by the user. See
        :func:`~pySW4.utils.geo.get_dem`.
    """

    latmin -= 1 if latmin < 0 else 0
    # latmax -= 1 if latmax < 0 else 0
    lonmin -= 1 if lonmin < 0 else 0
    # lonmax -= 1 if lonmax < 0 else 0

    latrange = range(int(latmin), int(latmax) + 1)
    lonrange = range(int(lonmin), int(lonmax) + 1)

    tile_filenames = []
    for lat in latrange:
        if lat < 0:
            N = 'S'
            lat = -1 * lat
        else:
            N = 'N'
        for lon in lonrange:
            if lon < 0:
                E = 'W'
                lon = -1 * lon
            else:
                E = 'E'

            basename = tilename.format(N, lat, E, lon)
            fullname = os.path.join(path, basename + '.zip')

            if verbose:
                print(fullname)

            tile_filenames += [fullname]

    return tile_filenames


def get_dem(path, lonmin, lonmax, latmin, latmax,
            tilename=tilename_template, rasterBand=1, fill_value=None,
            verbose=False):
    """
    This function reads ASTER-GDEM GeoTIFF tiles into memory, stitches
    them together if more than one file is read, and cuts to the
    desired extent given by `lonmin`, `lonmax`, `latmin`, `latmax` in
    decimal degrees.

    Parameters
    ----------
    path : str
        Path (relative or absolute) to where the ASTER-GDEM zipped tiles
        are stored.

    lonmin, lonmax : float
        The west- and east-most coordinate of the desired extent
        (may also be the xmin and xmax coordinate).

    latmin, latmax : float
        The south- and north-most coordinate of the desired extent
        (may also be the ymin and ymax coordinate).

    tilename : str
        String template of the tile names.

    rasterBand : int
        The band number to read from each tile (defaults to 1).

    fill_value : float
        Value to fill missing data or missing tiles. Defaults to the
        tile ``nodata`` attribute.

    verbose : bool
        Print some information on the process.

    Returns
    -------
    :class:`~pySW4.utils.geo.GeoTIFF`
        A populated :class:`~pySW4.utils.geo.GeoTIFF` instance with the
        stitched elevation data.
    """

    # get a list of tiles for reading
    tiles = _get_tiles(path, lonmin, lonmax, latmin, latmax, tilename)

    gdems = []
    _lonmin, _lonmax = lonmin, lonmax
    _latmin, _latmax = latmin, latmax
    # read all GDEMs into memory for stitching
    for tile in tiles:
        if verbose:
            print('Processing tile: {}...'.format(tile))
        try:
            tile_ = read_GeoTIFF(tile, rasterBand)
            if verbose:
                print(tile_)
            _lonmin = min(_lonmin, tile_.w)
            _lonmax = max(_lonmax, tile_.e)
            _latmin = min(_latmin, tile_.s)
            _latmax = max(_latmax, tile_.n)
            gdems += [tile_]
        except FileNotFoundError as e:
            warn(('{}: {}\n'
                  'Tile space is filled with the set fill_value...').format(
                repr(e), tile))

    # create a mosaicGDEM (a GeoTIFF class instance) which will eventually
    # hold the GDEM data within the specified extent in
    # lonmin, lonmax, latmin, latmax
    if len(gdems) is 1:
        mosaicGDEM = gdems[0]
    else:
        mosaicGDEM = GeoTIFF()

        mosaicGDEM.dx     = gdems[0].dx
        mosaicGDEM.dy     = gdems[0].dy
        mosaicGDEM.proj4  = gdems[0].proj4
        mosaicGDEM.nodata = gdems[0].nodata

        mosaicGDEM.w = _lonmin
        mosaicGDEM.n = _latmax

        _nx = int(0.5 + abs((_lonmax - _lonmin) / gdems[0].dx))
        _ny = int(0.5 + abs((_latmax - _latmin) / gdems[0].dy))

        # initialize the z array into which data from each
        # individual tile will be added
        mosaicGDEM.z = np.zeros((_ny, _nx))

        # stitching all gdems into one big mosaic
        for gdem in gdems:
            npw = int(0.5 + ((gdem.w - mosaicGDEM.w) / mosaicGDEM.dx))
            npe = int(0.5 + ((gdem.e - mosaicGDEM.w) / mosaicGDEM.dx))

            nps = int(0.5 + ((gdem.s - mosaicGDEM.n) / mosaicGDEM.dy))
            npn = int(0.5 + ((gdem.n - mosaicGDEM.n) / mosaicGDEM.dy))

            mosaicGDEM.z[npn: nps, npw: npe] = gdem.z

    mosaicGDEM.set_new_extent(lonmin, lonmax, latmin, latmax, fill_value)

    return mosaicGDEM


def xy2gridpoints(x, y, grid_shape, grid_extent):
    """
    Get the grid coordinates of the data coordinates in ``x``, ``y``
    """

    dx = np.abs(grid_extent[1] - grid_extent[0]) / grid_shape[0]
    dy = np.abs(grid_extent[3] - grid_extent[2]) / grid_shape[1]

    xgp = ((x - grid_extent[0]) / dx).astype(int)
    ygp = ((y - grid_extent[2]) / dy).astype(int)
    return xgp, ygp
