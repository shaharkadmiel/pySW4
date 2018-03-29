"""
Python module for handling DEM/GDEM/DTM and other raster data readable
with gdal.

.. module:: geo

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

from scipy import ndimage
import numpy as np
import sys
import os
import zipfile
import shutil
import copy
from warnings import warn

try:
    from osgeo import gdal, osr, gdal_array
except ImportError:
    warn('gdal not found, you will not be able to use the geo tools '
         ' in `pySW4.utils.geo` unless you install gdal.')

GDAL_INTERPOLATION = {'nearest' : gdal.GRA_NearestNeighbour,
                      'bilinear': gdal.GRA_Bilinear,
                      'cubic'   : gdal.GRA_Cubic,
                      'lanczos' : gdal.GRA_Lanczos}


class GeoTIFF():
    """
    Class for handling GeoTIFF files.

    .. note:: This class should be populated by
              :func:`~pySW4.utils.geo.read_GeoTIFF` or
              :func:`~pySW4.utils.geo.get_dem`.
    """

    def __init__(self):
        self.path = None
        self.name = None
        self.dtype = np.int16
        self.nodata = np.nan
        self.w = 0.
        self.e = 0.
        self.s = 0.
        self.n = 0.
        self.extent = (0, 0, 0, 0)
        self.proj4 = ''
        self.dx = 0.
        self.dy = 0.
        self.nx = 0
        self.ny = 0

        self.elev = np.array([])

    def _read(self, filename, rasterBand, verbose=False):
        """
        Private method for reading a GeoTIFF file.
        """
        try:
            self.path, self.name = os.path.split(filename)

            if verbose:
                print('Reading GeoTIFF file: %s' % filename)
                sys.stdout.flush()

            src_ds = gdal.Open(filename)
        except AttributeError:
            if verbose:
                print('Updating GeoTIFF file...')
                sys.stdout.flush()
            src_ds = filename

        self._src_ds = src_ds
        band = src_ds.GetRasterBand(rasterBand)
        self.dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)
        self.nodata = band.GetNoDataValue()
        self.geotransform = src_ds.GetGeoTransform()
        self.w = self.geotransform[0]
        self.n = self.geotransform[3]
        self.dx = self.geotransform[1]
        self.dy = self.geotransform[5]
        self.nx = band.XSize
        self.ny = band.YSize

        self.e = self.w + self.nx * self.dx
        self.s = self.n + self.ny * self.dy

        self.extent = (self.w, self.e, self.s, self.n)
        self.proj4 = osr.SpatialReference(self._src_ds.GetProjection())\
                        .ExportToProj4()

        self.elev = band.ReadAsArray()

    def _update(self):
        """
        Update GeoTIFF object by closing and reopening the dataset
        """
        pass

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

    def resample(self, by=None, to=None, order=0, keep_extent=False):
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

            - If ``by=1``, nothing happens.
            - If ``by<1``, the grid is sub-sampled by the factor.
            - if ``by>1``, the grid is super-sampled byt the factor.

        to : float
            The specified spacing to which the grid is resampled to.

        order : int
            The order of the spline interpolation, default is 0.
            The order has to be in the range 0-5.

            - 0 - nearest (fastest)
            - 1 - bi-linear
            - 3 - cubic (slower)
            - ...

        keep_extent : bool
            Resampling interpolates the data onto a new grid with the
            desired spacing. This may result in a change of the extent
            of the data. To keep the extent, set `keep_extent` to
            ``True``.

            **Note** that this may result in slightly different
            spacing than desired and more importantly, may cause a
            **discrepency** between `x` and `y` spacing.

        """

        if by and to:
            msg = 'Only `by` or `to` should be specified, not both.'
            raise ValueError(msg)

        if to:
            by = abs(self.dx / to)

        if by:
            to = abs(self.dx / by)

        self.elev = ndimage.zoom(self.elev, by, order=order)
        self.ny, self.nx = self.elev.shape
        if keep_extent:
            self.dy, self.dx = ((self.s - self.n) / (self.ny - 1),
                                (self.e - self.w) / (self.nx - 1))
        else:
            self.dy, self.dx = -to, to
            self.e = self.w + self.nx * self.dx
            self.s = self.n + self.ny * self.dy

            self.extent = (self.w, self.e, self.s, self.n)

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
            (lrx, lry, lrz) = tx.TransformPoint(
                                self.w + self.dx * self.nx,
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

    # def reproject(self, epsg=None, proj4=None, match=None,
    #               error_threshold=0.125, target_filename=None):
    #     """
    #     Reproject the data from the current projection to the specified
    #     target projection `epsg` or `proj4` or to `match` an
    #     existing GeoTIFF file.

    #     .. warning:: **Watch Out!** This operation is performed in place
    #                  on the actual data. The raw data will no longer be
    #                  accessible afterwards. To keep the original data,
    #                  use the :meth:`~pySW4.utils.geo.GeoTIFF.copy`
    #                  method to create a copy of the current object.

    #     Parameters
    #     ----------
    #     epsg : int
    #         The target projection EPSG code. See the
    #         `Geodetic Parameter Dataset Registry
    #         <http://www.epsg-registry.org/>`_ for more information.

    #     proj4 : str
    #         The target Proj4 string. If the EPSG code is unknown or a
    #         custom projection is required, a Proj4 string can be passed.
    #         See the `Proj4 <https://trac.osgeo.org/proj/wiki/GenParms>`_
    #         documentation for a list of general Proj4 parameters.

    #     match : str or :class:`~pySW4.utils.geo.GeoTIFF` instance
    #         Path (relative or absolute) to an existing GeoTIFF file or
    #         :class:`~pySW4.utils.geo.GeoTIFF` instance (already in
    #         memory) to match size and projection of. Current data is
    #         resampled to match the shape and number of pixels of the
    #         existing GeoTIFF file or object.
    #         It is assumed that both GeoTIFF objects cover the same
    #         extent.

    #     error_threshold : float
    #         Error threshold for transformation approximation in pixel
    #         units. (default is 0.125 for compatibility with  gdalwarp)

    #     target_filename : str
    #         If a target filename is given then the reprojected data is
    #         saved as target_filename and read into memory replacing
    #         the current data. This is faster than reprojecting and then
    #         saving. Otherwise the
    #         :meth:`~pySW4.utils.geo.GeoTIFF.write_GeoTIFF` method can be
    #         used to save the data at a later point if further
    #         manipulations are needed.

    #     Notes
    #     -----
    #     Examples of some EPSG codes and their equivalent Proj4 strings
    #     are::

    #         4326   -> '+proj=longlat +datum=WGS84 +no_defs'

    #         32636  -> '+proj=utm +zone=36 +datum=WGS84 +units=m +no_defs'

    #         102009 -> '+proj=lcc +lat_1=20 +lat_2=60 +lat_0=40
    #                    +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m
    #                    +no_defs'

    #     and so on ... See the `Geodetic Parameter Dataset Registry
    #     <http://www.epsg-registry.org/>`_ for more information.
    #     """
    #     if epsg and proj4 and match:
    #         msg = 'Only `epsg`, `proj4`, or `match` should be specified.'
    #         raise ValueError(msg)
    #     elif not epsg and not proj4 and not match:
    #         msg = '`epsg`, `proj4`, or `match` MUST be specified.'
    #         raise ValueError(msg)

    #     if not target_filename:
    #         target_filename = 'dst_temp.tif'

    #     dstSRS = osr.SpatialReference()

    #     if epsg or proj4:
    #         try:
    #             dstSRS.ImportFromEPSG(epsg)
    #         except TypeError:
    #             dstSRS.ImportFromProj4(proj4)

    #         temp_ds = gdal.AutoCreateWarpedVRT(self._src_ds,
    #                                            None,
    #                                            dstSRS.ExportToWkt(),
    #                                            gdal.GRA_NearestNeighbour,
    #                                            error_threshold)
    #         dst_ds = gdal.GetDriverByName('GTiff').CreateCopy(target_filename,
    #                                                           temp_ds)

    #     # isinstance(match, GeoTIFF)
    #     elif match:
    #         self.write_GeoTIFF('src_temp.tif')
    #         src_ds = gdal.Open('src_temp.tif')
    #         src_proj = src_ds.GetProjection()
    #         src_geotrans = src_ds.GetGeoTransform()

    #         if type(match) is str:
    #             match = read_GeoTIFF(match, 1)

    #         gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype)
    #         dstSRS.ImportFromProj4(match.proj4)
    #         dst_ds = gdal.GetDriverByName('GTiff').Create(target_filename,
    #                                                       match.nx,
    #                                                       match.ny,
    #                                                       1,
    #                                                       gdal_dtype)
    #         dst_ds.SetGeoTransform(match.geotransform)
    #         dst_ds.SetProjection(dstSRS.ExportToWkt())

    #         gdal.ReprojectImage(src_ds, dst_ds,
    #                             src_proj, dstSRS.ExportToWkt(),
    #                             gdal.GRA_NearestNeighbour,
    #                             error_threshold)
    #         os.remove('src_temp.tif')

    #     dst_ds = None
    #     self._read(target_filename, 1)
    #     try:
    #         os.remove('dst_temp.tif')
    #     except OSError:
    #         pass

    def keep(self, w, e, s, n):
        """
        Keep a subset array from a GeoTIFF file.

        .. warning:: **Watch Out!** This operation is performed in place
            on the actual data. The raw data will no longer be
            accessible afterwards. To keep the original data, use the
            :meth:`~pySW4.utils.geo.GeoTIFF.copy` method to create a
            copy of the current object.

        Parameters
        ----------
        w, e, s, n: float
            The west-, east-, south-, and north-most coordinate to keep
            (may also be the xmin, xmax, ymin, ymax coordinate).
        """

        if (self.w < w < self.e and
                self.w < e < self.e and
                self.s < s < self.n and
                self.s < n < self.n):

            x_start = int((w - self.w) / self.dx)
            x_stop  = int((e - self.w) / self.dx) + 1

            y_start = int((n - self.n) / self.dy)
            y_stop  = int((s - self.n) / self.dy) + 1

            # update class data
            self.e = self.w + x_stop  * self.dx
            self.w = self.w + x_start * self.dx
            self.s = self.n + y_stop  * self.dy
            self.n = self.n + y_start * self.dy

            self.extent = (self.w, self.e, self.s, self.n)
            self.elev = self.elev[y_start:y_stop, x_start:x_stop]

            self.nx = self.elev.shape[1]
            self.ny = self.elev.shape[0]
        else:
            msg = ('One or more of the coordinates given is out of bounds:\n'
                   '{} < `w` and `e` < {} and {} < `s` and `n` < {}')
            raise ValueError(msg.format(self.w, self.e, self.s, self.n))

    def elevation_profile(self, xs, ys):
        """
        Get the elevation values ar points in ``xs`` and ``ys``.
        """
        rows = []
        columns = []
        for x, y in zip(xs, ys):
            row = (x - self.w) / self.dx
            column = (self.n - y) / self.dy

            if row > self.elev.shape[1] or column > self.elev.shape[0]:
                continue
            # Add the point to our return array
            rows += [int(row)]
            columns += [int(column)]
        return self.elev[columns, rows]

    def get_intensity(self, azimuth=315., altitude=45.,
                      scale=None, smooth=None):
        """
        This is a method to create an intensity array that can be used to
        create a shaded relief map.

        .. note:: Since version 0.2.0 this method is not needed anymore
                  as new hillshade code has been implemented in the
                  :mod:`~pySW4.plotting.hillshade` module.

        Parameters
        ----------
        azimuth : float
            Direction of light source, degrees from north.

        altitude : float
            Height of light source, degrees above the horizon.

        scale : float
            Scaling value of the data.

        smooth : float
            Number of cells to average before intensity calculation.

        Returns
        -------
        a 2d :class:`~numpy.ndarray`
            Normalized 2d array with illumination data.
        """
        return calc_intensity(self.elev, azimuth, altitude, scale, smooth)

    def write_topo_file(self, filename):
        """
        Write an ASCII Lon, Lat, z file for SW4 topography.
        """

        header = '%d %d' % (self.nx, self.ny)
        lon, lat = self.xy2d()
        np.savetxt(filename, np.column_stack((lon.ravel(),
                                              lat.ravel(),
                                              self.elev.ravel())),
                   fmt='%f', header=header, comments='')

    def set_nodata(self, value):
        """
        Replace the current ``nodata`` value with the new `value`.
        """

        try:
            if np.isnan(self.elev).any():
                self.elev[np.isnan(self.elev)] = value
            else:
                self.elev[self.elev == self.nodata] = value

            self.nodata = value
        except ValueError:
            self.elev = self.elev.astype(np.float32)
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

        save_GeoTIFF(filename, self.elev, self.w, self.n,
                     self.dx, self.dy, proj4=self.proj4,
                     dtype=self.dtype, nodata=self.nodata,
                     rasterBand=1)

    def copy(self):
        """
        Return a deepcopy of the GeoTIFF object.
        """
        new = GeoTIFF()
        new.path = copy.deepcopy(self.path)
        new.name = copy.deepcopy(self.name)
        new.dtype = copy.deepcopy(self.dtype)
        new.nodata = copy.deepcopy(self.nodata)
        new.w = copy.deepcopy(self.w)
        new.e = copy.deepcopy(self.e)
        new.s = copy.deepcopy(self.s)
        new.n = copy.deepcopy(self.n)
        new.extent = copy.deepcopy(self.extent)
        new.proj4 = copy.deepcopy(self.proj4)
        new.dx = copy.deepcopy(self.dx)
        new.dy = copy.deepcopy(self.dy)
        new.nx = copy.deepcopy(self.nx)
        new.ny = copy.deepcopy(self.ny)
        new.elev = copy.deepcopy(self.elev)
        try:
            new._src_ds = gdal.Open(os.path.join(self.path, self.name))
        except AttributeError:
            new._src_ds = None
        return new


def calc_intensity(relief, azimuth=315., altitude=45.,
                   scale=None, smooth=None):
    """
    This is a method to create an intensity array that can be used to
    create a shaded relief map.

    .. note:: Since version 0.2.0 this function is not needed anymore as
              new hillshade code has been implemented in the
              :mod:`~pySW4.plotting.hillshade` module.

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

    Returns
    -------
    a 2d :class:`~numpy.ndarray`
        Normalized 2d array with illumination data.
        Same size as ``relief``.
    """

    relief = relief.copy()

    if scale is not None:
        relief *= scale
    if smooth:
        relief = ndimage.uniform_filter(relief, size=smooth)

    dx, dy = np.gradient(relief)

    slope = 0.5 * np.pi - np.arctan(np.hypot(dx, dy))

    aspect = np.arctan2(dx, dy)

    altitude = np.radians(altitude)
    azimuth = np.radians(180. - azimuth)

    intensity = (np.sin(altitude) * np.sin(slope)
                 + np.cos(altitude) * np.cos(slope)
                 * np.cos(-azimuth - np.pi - aspect))

    return (intensity - intensity.min()) / intensity.ptp()


def read_GeoTIFF(filename=None, rasterBand=1, verbose=False):
    """
    Reads a single GeoTIFF file and returns a
    :class:`~pySW4.utils.geo.GeoTIFF` class instance. If `filename`
    is None, an empty :class:`~pySW4.utils.geo.GeoTIFF` object is
    returnd.

    Returns
    -------
    :class:`~pySW4.utils.geo.GeoTIFF`
        A populated (or empty) :class:`~pySW4.utils.geo.GeoTIFF`
        instance with the elevation data.
    """

    tif = GeoTIFF()

    if not filename:  # return an empty GeoTIFF object
        return tif
    else:
        tif._read(filename, rasterBand, verbose)

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


def _get_tiles(path, lonmin, lonmax, latmin, latmax,
               tilename='ASTGTM2_%s%02d%s%03d', verbose=False):
    """
    This is a helper function for :func:`~pySW4.utils.geo.get_dem`
    function.

    It extracts the GeoTIFF tiles from the zipfile downloaded from the
    ASTER-GDEM dowonload site needed to stitch an elevation model
    with the extent given by `lonmin`, `lonmax`, `latmin`, `latmax` in
    decimal degrees and returns a list of paths pointing to these tiles.

    .. note:: This function sould not be used directly by the user. See
              :func:`~pySW4.utils.geo.get_dem`.
    """

    tiles = []
    tile_index = []
    latrange = range(int(latmin), int(latmax + 1))
    lonrange = range(int(lonmin), int(lonmax + 1))
    for i in latrange:
        if i < 0:
            N = 'S'
            i = -1 * i + 1
        else:
            N = 'N'
        for j in lonrange:
            if j < 0:
                E = 'W'
                j = -1 * j + 1
            else:
                E = 'E'
            basename = tilename % (N, i, E, j)
            fullname = os.path.join(path, basename)

            # make tifs directory on current path
            tif_path = './tifs'
            if not os.path.exists(tif_path):
                os.makedirs('tifs')

            f = os.path.join(tif_path, basename + '_dem.tif')

            if verbose:
                print(f)

            if os.path.exists(f):
                tiles += [f]
                continue
            else:
                if not os.path.exists(fullname + '.zip'):
                    msg = ('Warning! missing file {}. '
                           'Replacing tile with zeros.')
                    warn(msg.format(fullname + '.zip'))
                    sys.stdout.flush()
                    tiles += [basename]
                    continue

                if verbose:
                    print('Extracting tiles to %s' % pwd)

                with zipfile.ZipFile(fullname + '.zip') as zf:
                    for item in zf.namelist():
                        if item.endswith('_dem.tif'):
                            dir_, tif_ = os.path.split(item)
                            zf.extract(item, tif_path)
                            if dir_:
                                shutil.move(
                                    os.path.join(tif_path, item), tif_path)
                                shutil.rmtree(os.path.join(tif_path, dir_))
                tiles += [f]

    return tiles


def get_dem(path, lonmin, lonmax, latmin, latmax,
            tilename='ASTGTM2_%s%02d%s%03d', rasterBand=1,
            dx=0.000277777777778, dy=-0.000277777777778,
            keep_tiles=False):
    """
    This function reads relevant ASTER-GDEM GeoTIFF tiles into memory,
    stitchs them together if more than one file is read, and cuts to the
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
        Wildcard string of the tile names.

    rasterBand : int
        The band number to read from each tile (defaults to 1).

    dx, dy : float
        Pixel size of data in the x and y direction, positive in the
        East and North directions.

    keep_tiles : bool
        Zipped tiles get extracted to the current working directory and
        are removed once the stitching is complete. If you are
        interested in keeping the tiles in the current working directory
        set to ``True``.

    Returns
    -------
    :class:`~pySW4.utils.geo.GeoTIFF`
        A populated :class:`~pySW4.utils.geo.GeoTIFF` instance with the
        stitched elevation data.
    """

    # get a list of tiles for reading
    tiles = _get_tiles(path, lonmin, lonmax, latmin, latmax, tilename)

    gdems = []
    emptys = []
    # read all GDEMs into memory to stitch together and sample
    for tile in tiles:
        try:
            gdems += [read_GeoTIFF(tile, rasterBand)]

        except RuntimeError:
            empty = GeoTIFF()
            empty.name = tile
            empty.dtype = np.int16
            empty.nodata = np.nan

            try:
                coordinates = tile.split('N')[-1]
            except ValueError:
                coordinates = tile.split('S')[-1]

            try:
                n, w = coordinates.split('E')
            except:
                n, w = coordinates.split('W')
            empty.w = float(w) - 0.5 * dx
            empty.e = empty.w + 1 + dx
            empty.n = float(n) + 1 - 0.5 * dy
            empty.s = empty.n - 1 + dy

            empty.extent = (empty.w, empty.e, empty.s, empty.n)

            empty.dx = dx
            empty.dy = dy
            empty.nx = abs(int(round(1 / dx))) + 1
            empty.ny = abs(int(round(1 / dy))) + 1

            empty.elev = np.zeros((empty.ny, empty.nx))
            emptys += [empty]

    # remove the tiles after reading them
    if not keep_tiles:
        shutil.rmtree('./tifs')
    gdems += emptys

    # create a mosaicGDEM (a GeoTIFF class instance) which will eventually
    # hold the GDEM data within the specified extent in
    # lonmin, lonmax, latmin, latmax
    if len(gdems) is 1:
        mosaicGDEM = gdems[0]
        elev = mosaicGDEM.elev
    else:
        mosaicGDEM = GeoTIFF()

        mosaicGDEM.dx     = gdems[0].dx
        mosaicGDEM.dy     = gdems[0].dy
        mosaicGDEM.proj4  = gdems[0].proj4
        mosaicGDEM.dtype  = gdems[0].dtype
        mosaicGDEM.nodata = gdems[0].nodata

        mosaicGDEM.w = min([gdem.w for gdem in gdems])
        mosaicGDEM.e = max([gdem.e for gdem in gdems])
        mosaicGDEM.s = min([gdem.s for gdem in gdems])
        mosaicGDEM.n = max([gdem.n for gdem in gdems])
        mosaicGDEM.nx = abs(int((mosaicGDEM.e - mosaicGDEM.w)
                                / mosaicGDEM.dx)) + 1
        mosaicGDEM.ny = abs(int((mosaicGDEM.n - mosaicGDEM.s)
                                / mosaicGDEM.dy)) + 1

        # initialize the elev array into which data from each
        # individual tile will be added
        elev = np.zeros((mosaicGDEM.ny, mosaicGDEM.nx))

        # stitching all gdems into one big mosaic
        for gdem in gdems:
            x_start = int((gdem.w - mosaicGDEM.w) / mosaicGDEM.dx)
            x_stop = x_start + gdem.nx

            y_start = int((gdem.n - mosaicGDEM.n) / mosaicGDEM.dy)
            y_stop = y_start + gdem.ny

            elev[y_start:y_stop, x_start:x_stop] = gdem.elev

    mosaicGDEM.elev = elev
    mosaicGDEM.keep(lonmin, lonmax, latmin, latmax)

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
