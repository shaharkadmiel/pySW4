"""
- geo.py -

Python module for handling DEM/GDEM/DTM and other raster data readable with
gdal.

By: Shahar Shani-Kadmiel, May 2014, kadmiel@post.bgu.ac.il

"""
from __future__ import absolute_import, print_function, division

from scipy import ndimage
import numpy as np
import sys, os, zipfile, shutil, copy
import warnings

try:
    from osgeo import gdal, osr, gdal_array
except ImportError:
    warnings.warn('gdal not found, you will not be able to use the geo tools'\
                  ' in `pySW4.utils.geo` unless you install gdal.')


class GeoTIFF(object):
    """class for GeoTIFF files"""

    def __init__(self):
        self.path = None
        self.name = None
        self.dtype = np.int16
        self.nodata = np.nan
        self.w = 0.
        self.e = 0.
        self.s = 0.
        self.n = 0.
        self.extent = (0,0,0,0)
        self.proj4 = ''
        self.dx = 0.
        self.dy = 0.
        self.nx = 0
        self.ny = 0

        self.elev = np.array([])


    def _read(self, filename, rasterBand, verbose=False):
        self.path, self.name = os.path.split(filename)

        if verbose:
            print('Reading GeoTIFF file: %s' %filename)
            sys.stdout.flush()

        src_ds = gdal.Open(filename)

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

        self.e = self.w + (self.nx)*self.dx
        self.s = self.n + (self.ny)*self.dy

        self.extent = (self.w, self.e, self.s, self.n)
        self.proj4 = osr.SpatialReference(self._src_ds.GetProjection())\
                     .ExportToProj4()

        self.elev = band.ReadAsArray()


    def __str__(self):
        string = '\nGeoTIFF info:\n'
        if self.name:
            string += 'name: %s\n' %self.name
        string += 'west: %s\n' %self.w
        string += 'east: %s\n' %self.e
        string += 'south: %s\n' %self.s
        string += 'north: %s\n' %self.n
        string += 'x pixel size: %s\n' %self.dx
        string += 'y pixel size: %s\n' %self.dy
        string += '# of x pixels: %s\n' %self.nx
        string += '# of y pixels: %s\n' %self.ny
        string += 'no data value: %s\n' %self.nodata
        string += 'data type: %s\n' %self.dtype
        string += 'proj4: %s\n' %self.proj4

        return string


    @property
    def x(self):
        return np.linspace(self.w, self.e, self.nx)


    @property
    def y(self):
        return np.linspace(self.n, self.s, self.ny)


    @property
    def xy2d(self):
        return np.meshgrid(self.x(),self.y())


    def resample(self, by=None, to=None, order=3):
        """Method to resample the data either *by* a factor or
        *to* the specified spacing. Uses `scipy.ndimage.zoom`.
        ** Watch Out **
        This operation is performed in place on the actual data. The
        raw data will no longer be accessible afterwards. To keep the
        original data, use the copy method `GeoTIFF.copy()` to create
        a copy of the current object.
        Params:
        -------
        by: if 1, nothing happens
            if < 1, the grid is sub-sampled by the factor
            if > 1, the grid is super-sampled byt the factor
        to: the specified spacing to which the grid is resampled to.
        order: int,
                The order of the spline interpolation, default is 3.
                The order has to be in the range 0-5
                0 - nearest
                1 - bi-linear
                3 - cubic
                ...
        """

        if by and to:
            msg = 'Only `by` or `to` should be specified, not both.'
            raise ValueError(msg)

        if to:
            by = abs(self.dx/to)

        self.elev = ndimage.zoom(self.elev,by,order=order)
        self.ny, self.nx = self.elev.shape
        self.dy, self.dx = (self.s-self.n)/(self.ny-1),\
                               (self.e-self.w)/(self.nx-1)


    def reproject(self, epsg=None, proj4=None, match=None,
                  error_threshold=0.125, target_filename=None):
        """Reproject the data from the current projection to the specified
        target projection `epsg` or `proj4` or to match an existing GeoTIFF
        file `match`.
        ** Watch Out **
        This operation is performed in place on the actual data. The
        raw data will no longer be accessible afterwards. To keep the
        original data, use the copy method `GeoTIFF.copy()` to create
        a copy of the current object.
        Params:
        -------
        epsg : target EPSG code.
            a reference number to the EPSG Geodetic Parameter Dataset registry.
            examples of some EPSG codes and their equivalent Proj4 strings are:
            4326   -> +proj=longlat +datum=WGS84 +no_defs
            32636  -> +proj=utm +zone=36 +datum=WGS84 +units=m +no_defs
            102009 -> +proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96
                      +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs
            ...
        proj4 : target Proj4 string
            if the EPSG code is unknown or a custom projection is required,
            a Proj4 string can be passed.
            See https://trac.osgeo.org/proj/wiki/GenParms for a list of
            general Proj4 parameters.
        match : filename of an existing GeoTIFF file or object (already in
            memory) to match size and projection. current data is resampled
            to match the shape and number of pixels of the existing GeoTIFF
            file or object.
            It is assumed that both data cover the same extent.
        error_threshold : error threshold for transformation approximation
            in pixel units. (default is 0.125 as in gdalwarp)
        target_filename : if a target filename is given then the reprojected
            data is saved as target_filename and read into memory replacing
            the current data. This is faster than reprojecting and then
            saving. Otherwise the `GeoTIFF.write_GeoTIFF` method can be used
            to save the data at a later point if further manipulations are
            needed.
        """

        if epsg and proj4 and match:
            msg = 'Only `epsg`, `proj4`, or `match` should be specified.'
            raise ValueError(msg)
        elif not epsg and not proj4 and not match:
            msg = '`epsg`, `proj4`, or `match` MUST be specified.'
            raise ValueError(msg)

        if not target_filename:
            target_filename = 'dst_temp.tif'

        dstSRS = osr.SpatialReference()

        if epsg or proj4:
            try:
                dstSRS.ImportFromEPSG(epsg)
            except TypeError:
                dstSRS.ImportFromProj4(proj4)

            temp_ds = gdal.AutoCreateWarpedVRT(self._src_ds,
                                               None,
                                               dstSRS.ExportToWkt(),
                                               gdal.GRA_NearestNeighbour,
                                               error_threshold)
            dst_ds = gdal.GetDriverByName('GTiff').CreateCopy(target_filename,
                                                              temp_ds)

        #isinstance(match, GeoTIFF)
        elif match:
            self.write_GeoTIFF('src_temp.tif')
            src_ds = gdal.Open('src_temp.tif')
            src_proj = src_ds.GetProjection()
            src_geotrans = src_ds.GetGeoTransform()

            if type(match) is str:
                match = read_GeoTIFF(match,1)

            gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(self.dtype)
            dstSRS.ImportFromProj4(match.proj4)
            dst_ds = gdal.GetDriverByName('GTiff').Create(target_filename,
                                                          match.nx,
                                                          match.ny,
                                                          1,
                                                          gdal_dtype)
            dst_ds.SetGeoTransform(match.geotransform)
            dst_ds.SetProjection(dstSRS.ExportToWkt())

            gdal.ReprojectImage(src_ds, dst_ds,
                                src_proj, dstSRS.ExportToWkt(),
                                gdal.GRA_NearestNeighbour,
                                error_threshold)
            os.remove('src_temp.tif')

        dst_ds = None
        self._read(target_filename, 1)
        try:
            os.remove('dst_temp.tif')
        except OSError:
            pass


    def keep(self, w, e, s, n):
        """
        Keep a subset array from a GeoTIFF file.
        ** Watch Out **
        This operation is performed in place on the actual data. The
        raw data will no longer be accessible afterwards. To keep the
        original data, use the copy method `GeoTIFF.copy()` to create
        a copy of the current object.
        """

        if not (self.w < w < self.e or
                self.w < e < self.e or
                self.s < s < self.n or
                self.s < n < self.n):
            msg = 'One or more of the coordinates given is out of bounds:\n'\
                  '{} < `w` and `e` < {} and {} < `s` and `n` < {}'
            raise ValueError(msg.format(self.w, self.e, self.s, self.n))

        x_start = int((w - self.w)/self.dx)
        x_stop  = int((e - self.w)/self.dx)+1

        y_start = int((n - self.n)/self.dy)
        y_stop  = int((s - self.n)/self.dy)+1

        # update class data
        self.e = self.w + x_stop  * self.dx
        self.w = self.w + x_start * self.dx
        self.s = self.n + y_stop  * self.dy
        self.n = self.n + y_start * self.dy

        self.extent = (self.w, self.e, self.s, self.n)
        self.elev = self.elev[y_start:y_stop, x_start:x_stop]

        self.nx = self.elev.shape[1]
        self.ny = self.elev.shape[0]


    def elevation_profile(self, xs, ys):

        rows = []; columns = []
        for x, y in zip(xs, ys):
            row = (x-self.w)/self.dx
            column = (self.n-y)/self.dy

            if row > self.elev.shape[1] or column > self.elev.shape[0]:
                continue
            # Add the point to our return array
            rows += [int(row)]
            columns += [int(column)]
        return self.elev[columns, rows]


    def get_intensity(self, azimuth=315., altitude=45.,
                      scale=None, smooth=None):
        """This is a method to create an intensity array that can be used to
        create a shaded relief map.
        Params:
        -------
        azimuth: direction of light source, degrees from north
        altitude: height of light source, degrees above the horizon
        scale: scaling value of the data
        smooth: number of cells to average before intensity calculation
        """
        return calc_intensity(self.elev, azimuth, altitude, scale, smooth)


    def write_topo_file(self, filename):
        """Write an ASCII Lon, Lat, z file for SW4 topography"""

        header = '%d %d' %(self.nx, self.ny)
        lon,lat = self.xy2d()
        np.savetxt(filename, np.column_stack((lon.ravel(),
                                              lat.ravel(),
                                              self.elev.ravel())),
                   fmt='%f', header=header, comments='')


    def write_GeoTIFF(self, filename, nodata=None):
        """Write a GeoTIFF file."""

        if not nodata:
            nodata = self.nodata
            if not nodata:
                nodata = np.nan

        save_GeoTIFF(filename, self.elev, self.w, self.n,
                     self.dx, self.dy, proj4=self.proj4,
                     dtype=self.dtype, nodata=nodata,
                     rasterBand=1)

    def copy(self):
        """
        Return a deepcopy of the GeoTIFF object.
        """
        return copy.deepcopy(self)


def calc_intensity(relief, azimuth=315., altitude=45.,
                   scale=None, smooth=None):
    """This is a method to create an intensity array that can be used to
    create a shaded relief map.
    Params:
    --------
    relief : 2d array of topography or other data to calculate intensity from
    azimuth : direction of light source, degrees from north
    altitude : height of light source, degrees above the horizon
    scale : scaling value of the data, higher number higher gradient
    smooth : number of cells to average before intensity calculation
    Returns:
    ---------
    2d array, same size as `relief`
    """

    relief = relief.copy()

    if scale is not None:
        relief *= scale
    if smooth:
        relief = ndimage.uniform_filter(relief, size=smooth)

    dx, dy = np.gradient(relief)

    slope = 0.5*np.pi - np.arctan(np.hypot(dx, dy))

    aspect = np.arctan2(dx, dy)

    altitude = np.radians(altitude)
    azimuth = np.radians(180. - azimuth)

    intensity = (np.sin(altitude) * np.sin(slope) +
                 np.cos(altitude) * np.cos(slope) *
                 np.cos(-azimuth - np.pi - aspect))

    return (intensity - intensity.min())/intensity.ptp()


def read_GeoTIFF(filename=None, rasterBand=1, verbose=False):
    """Reads GeoTIFF_file and return a GeoTIFF class instance.
    If `filename` is None, an empty GeoTIFF object is returnd."""

    tif = GeoTIFF()

    if not filename: # return an empty GeoTIFF object
        return tif
    else:
        tif._read(filename, rasterBand, verbose)

    return tif


def save_GeoTIFF(filename, data, tlx, tly, dx, dy,
                 epsg=None, proj4=None,
                 dtype=np.int16, nodata=np.nan, rasterBand=1):
    """Save data at a known projection as a GeoTIFF file.
    Params:
    -------
    filename : name of the output GeoTIFF file
    data : a 2d array of data to write to file
    tlx : top-left x (usually West-) coordinate of data
    tly : top-left y (usually North-) coordinate of data
    dx : pixel size of data in the x direction, positive in the East direction
    dy : pixel size of data in the y direction, positive in the North direction
    epsg : a reference number to the EPSG Geodetic Parameter Dataset registry.
        examples of some EPSG codes and their equivalent Proj4 strings are:
        4326   -> +proj=longlat +datum=WGS84 +no_defs
        32636  -> +proj=utm +zone=36 +datum=WGS84 +units=m +no_defs
        102009 -> +proj=lcc +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96
                  +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs
        ...
    proj4 : if the EPSG code is unknown or a custom projection is required,
        a Proj4 string can be passed.
        See https://trac.osgeo.org/proj/wiki/GenParms for a list of general
        Proj4 parameters.
    dtype : one of following dtypes should be used `numpy.int16` (default),
        `numpy.int32`,`numpy.float32`, or `numpy.float64`.
        * Note that `float` is not the same as `numpy.float32` *
    nodata : set the no data value in the GeoTIFF file (default is
        `numpy.nan`). Other options are -9999, -12345 or any other
        value that suits your purpose.
    rasterband : the band number to write. (default is 1)
    """

    if epsg and proj4:
        msg = 'Only `epsg` or `proj4` should be specified, not both.'
        raise ValueError(msg)
    elif not epsg and not proj4:
        msg = '`epsg` or `proj4` MUST be specified.'
        raise ValueError(msg)

    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dtype)
    if not gdal_dtype:
        msg = 'Illegal DataType {}. `dtype` can be one of `numpy.int16`, '\
              '`numpy.int32`, `numpy.float32`, or `numpy.float64`.'
        raise TypeError(msg.format(dtype))

    dst_ds = gdal.GetDriverByName('GTiff').Create(filename,
                                                  data.shape[1],
                                                  data.shape[0],
                                                  1,
                                                  gdal_dtype)
    dst_ds.SetGeoTransform((tlx,dx,0, tly,0,dy))

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
               tilename='ASTGTM2_N%02dE%03d', verbose=False):
    """This is a helper function for `get_dem`.
    It extracts the GeoTIFF tiles needed to stitch an elevation model
    with the extent given by lonmin, lonmax, latmin, latmax
    in decimal degrees and
    returns a list pointing to these tiles.
    """

    tiles = []; tile_index = []
    latrange = range(int(latmin), int(latmax+1))
    lonrange = range(int(lonmin), int(lonmax+1))
    for i in latrange:
        for j in lonrange:
            basename = tilename %(i,j)
            fullname = os.path.join(path,basename)
            pwd = os.path.abspath('./')

            f = os.path.join(pwd,basename,basename + '_dem.tif')

            if verbose:
                print(f)

            if os.path.exists(f):
                tiles += [f]
                continue
            else:
                if not os.path.exists(fullname + '.zip'):
                    msg = 'Warning! missing file {}. '\
                          'Replacing tile with zeros.'
                    warn(msg.format(fullname + '.zip'))
                    sys.stdout.flush()
                    tiles += [basename]
                    continue

                if verbose:
                    print('Extracting tiles to %s' %pwd)

                with zipfile.ZipFile(fullname + '.zip') as zf:
                    zf.extract(os.path.join(basename,basename+'_dem.tif'))
                tiles += [f]

    return tiles


def get_dem(path, lonmin, lonmax, latmin, latmax,
            tilename='ASTGTM2_N%02dE%03d', rasterBand=1,
            dx=0.000277777777778, dy=-0.000277777777778,
            keep_tiles=False):
    """This function reads all relevant AsterGDEM GeoTIFF
    tiles into memory, stitching them together if needed,
    and cutting to the extent given by lonmin, lonmax,
    latmin, latmax in decimal degrees.
    Returns a GeoTIFF class instance"""


    # get a list of tiles for reading
    tiles = _get_tiles(path, lonmin, lonmax, latmin, latmax, tilename)

    gdems = []; emptys = []
    # read all GDEMs into memory to stitch together and sample
    for tile in tiles:
        try:
            gdems += [read_GeoTIFF(tile, rasterBand)]

            # remove the tiles after reading them
            if not keep_tiles:
                shutil.rmtree(os.path.split(tile)[0])

        except RuntimeError:
            empty = GeoTIFF()
            empty.name = tile
            empty.dtype = np.int16
            empty.nodata = np.nan

            coordinates = tile.split('N')[-1]
            n,w = coordinates.split('E')
            empty.w = float(w) - 0.5*dx
            empty.e = empty.w + 1 + dx
            empty.n = float(n) + 1 - 0.5*dy
            empty.s = empty.n - 1 + dy

            empty.extent = (empty.w,empty.e,empty.s,empty.n)

            empty.dx = dx
            empty.dy = dy
            empty.nx = abs(int(round(1/dx))) + 1
            empty.ny = abs(int(round(1/dy))) + 1

            empty.elev = np.zeros((empty.ny,empty.nx))
            emptys += [empty]

    gdems += emptys

    # create a mosaicGDEM (a GeoTIFF class instance) which will eventually
    # hold the GDEM data within the specified extent in
    # lonmin, lonmax, latmin, latmax
    if len(gdems) is 1:
        mosaicGDEM = gdems[0]
        elev = mosaicGDEM.elev
    else:
        mosaicGDEM = GeoTIFF()

        mosaicGDEM.dx   = gdems[0].dx
        mosaicGDEM.dy   = gdems[0].dy
        mosaicGDEM.proj4  = gdems[0].proj4
        mosaicGDEM.dtype  = gdems[0].dtype
        mosaicGDEM.nodata = gdems[0].nodata

        mosaicGDEM.w = min([gdem.w for gdem in gdems])
        mosaicGDEM.e = max([gdem.e for gdem in gdems])
        mosaicGDEM.s = min([gdem.s for gdem in gdems])
        mosaicGDEM.n = max([gdem.n for gdem in gdems])
        mosaicGDEM.nx = abs(int((mosaicGDEM.e - mosaicGDEM.w) /
                                  mosaicGDEM.dx)) + 1
        mosaicGDEM.ny = abs(int((mosaicGDEM.n - mosaicGDEM.s) /
                                  mosaicGDEM.dy)) + 1

        # initialize the elev array into which data from each
        # individual tile will be added
        elev = np.zeros((mosaicGDEM.ny, mosaicGDEM.nx))

        # stitching all gdems into one big mosaic
        for gdem in gdems:
            x_start = int((gdem.w - mosaicGDEM.w)/mosaicGDEM.dx)
            x_stop = x_start + gdem.nx

            y_start = int((gdem.n - mosaicGDEM.n)/mosaicGDEM.dy)
            y_stop = y_start + gdem.ny


            elev[y_start:y_stop, x_start:x_stop] = gdem.elev


    # subsample the big mosaic to the specified extent in
    # lonmin, lonmax, latmin, latmax
    # and update the class data
    x_start = abs(int((lonmin - mosaicGDEM.w)/mosaicGDEM.dx))
    x_stop  = abs(int((lonmax - mosaicGDEM.w)/mosaicGDEM.dx)) + 1

    y_start = abs(int((mosaicGDEM.n - latmax)/mosaicGDEM.dy))
    y_stop  = abs(int((mosaicGDEM.n - latmin)/mosaicGDEM.dy)) + 1

    # update class data
    mosaicGDEM.e = mosaicGDEM.w + x_stop*mosaicGDEM.dx
    mosaicGDEM.w = mosaicGDEM.w + x_start*mosaicGDEM.dx
    mosaicGDEM.s = mosaicGDEM.n + y_stop*mosaicGDEM.dy
    mosaicGDEM.n = mosaicGDEM.n + y_start*mosaicGDEM.dy

    mosaicGDEM.extent = (mosaicGDEM.w, mosaicGDEM.e,
                         mosaicGDEM.s, mosaicGDEM.n)
    mosaicGDEM.elev = elev[y_start:y_stop, x_start:x_stop]

    mosaicGDEM.nx = mosaicGDEM.elev.shape[1]
    mosaicGDEM.ny = mosaicGDEM.elev.shape[0]

    return mosaicGDEM


def xy2gridpoints(x,y,grid_shape,grid_extent):
    """get the grid coordinates of the data coordinates in `x`, `y`"""

    dx = np.abs(grid_extent[1]-grid_extent[0])/grid_shape[0]
    dy = np.abs(grid_extent[3]-grid_extent[2])/grid_shape[1]

    xgp = ((x - grid_extent[0])/dx).astype(int)
    ygp = ((y - grid_extent[2])/dy).astype(int)
    return xgp, ygp

