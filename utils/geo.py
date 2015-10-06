"""
- geo.py -

Python module for handling DEM/GDEM/DTM and other raster data readable with
gdal.

By: Shahar Shani-Kadmiel, May 2014, kadmiel@post.bgu.ac.il

"""

from scipy import ndimage
from scipy.misc import imresize
import numpy as np
import sys, os, zipfile, shutil
import osgeo.gdal as gdal, osr

def calc_intensity(relief, azimuth=315., altitude=45., scale=None, smooth=None):
    """Calculate intesnity array

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


def xy2gridpoints(x,y,grid_shape,grid_extent):
    """get the grid coordinates of the data coordinates in `x`, `y`"""

    dx = np.abs(grid_extent[1]-grid_extent[0])/grid_shape[0]
    dy = np.abs(grid_extent[3]-grid_extent[2])/grid_shape[1]

    xgp = ((x - grid_extent[0])/dx).astype(int)
    ygp = ((y - grid_extent[2])/dy).astype(int)
    return xgp, ygp


def get_tiles(path, lonmin, lonmax, latmin, latmax, tilename='ASTGTM2_N%02dE%03d'):
    """This function is a helper function for asterGDEM.
    It extracts the GeoTIFF tiles needed to mosaic an elevation model
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
            print f
            if os.path.exists(f):
                tiles += [f]
                continue
            else:
                if not os.path.exists(fullname + '.zip'):
                    print '** Warning! missing file %s **' %(fullname + '.zip')
                    continue

                print 'Extracting tiles to %s' %pwd
                with zipfile.ZipFile(fullname + '.zip') as zf:
                    zf.extract(os.path.join(basename,basename+'_dem.tif'))
                tiles += [f]

    return tiles

def read_GeoTIFF(GeoTIFF_file=None, rasterBand=1):
    """Reads GeoTIFF_file and return a GeoTIFF calss instance"""

    tif = GeoTIFF()

    if GeoTIFF_file is None: # return an empty GeoTIFF object
        return tif

    tif.path, tif.name = os.path.split(GeoTIFF_file)

    print 'Reading GeoTIFF file: %s' %GeoTIFF_file
    sys.stdout.flush()

    dataset = gdal.Open(GeoTIFF_file)
    if dataset is None:
        print 'Error: Unable to read GeoTIFF file: %s' %GeoTIFF_file
        return

    band = dataset.GetRasterBand(rasterBand)
    geotransform = dataset.GetGeoTransform()
    tif.w = geotransform[0]
    tif.n = geotransform[3]
    tif.dlon = geotransform[1]
    tif.dlat = geotransform[5]
    tif.nlon = band.XSize
    tif.nlat = band.YSize

    tif.e = tif.w + (tif.nlon-1)*tif.dlon
    tif.s = tif.n + (tif.nlat-1)*tif.dlat

    tif.extent = (tif.w, tif.e, tif.s, tif.n)

    tif.elev = band.ReadAsArray()

    return tif

def write_GeoTIFF(GeoTIFF_file, data, tlx, tly, dx, dy, epsg=4326,
                  nodata=np.nan, rasterBand=1):
    """Write a GeoTIFF file.
    """

    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(GeoTIFF_file,data.shape[1],data.shape[0],1,gdal.GDT_Float32)
    dst.SetGeoTransform((tlx,dx,0, tly,0,-dy))

    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromEPSG(epsg)
    dst.SetProjection(dstSRS.ExportToWkt())

    dstband = dst.GetRasterBand(rasterBand)
    dstband.WriteArray(data)
    dstband.SetNoDataValue(nodata)

    dstband.FlushCache()
    dst = None

def asterGDEM(path, lonmin, lonmax, latmin, latmax,
              tilename='ASTGTM2_N%02dE%03d', rasterBand=1,
              keep_tiles=False):
    """This function handles reading all the relevant AsterGDEM GeoTIFF
    tiles into memory, stitching them together if needed, and cutting
    to the extent given by lonmin, lonmax, latmin, latmax
    in decimal degrees.

    Returns a GeoTIFF class instance"""


    # get a list of tiles for reading
    tiles = get_tiles(path, lonmin, lonmax, latmin, latmax, tilename)

    gdems = []
    # read all GDEMs into memory to stitch together and sample
    for tile in tiles:
        gdems += [read_GeoTIFF(tile, rasterBand)]
        # remove the tiles after reading them
        if keep_tiles is False:
            shutil.rmtree(os.path.split(tile)[0])

    # create a mosaicGDEM (a GeoTIFF class instance) which will eventually
    # hold the GDEM data within the specified extent in
    # lonmin, lonmax, latmin, latmax
    if len(gdems) is 1:
        mosaicGDEM = gdems[0]
        elev = gdems[0].elev
        mosaicGDEM.dlon = abs(gdems[0].dlon)
        mosaicGDEM.dlat = abs(gdems[0].dlat)
    else:
        mosaicGDEM = GeoTIFF()

        mosaicGDEM.w = min([gdem.w for gdem in gdems])
        mosaicGDEM.e = max([gdem.e for gdem in gdems])
        mosaicGDEM.s = min([gdem.s for gdem in gdems])
        mosaicGDEM.n = max([gdem.n for gdem in gdems])
        mosaicGDEM.dlon = abs(gdems[0].dlon)
        mosaicGDEM.dlat = abs(gdems[0].dlat)
        mosaicGDEM.nlon = int((mosaicGDEM.e - mosaicGDEM.w) /
                                  mosaicGDEM.dlon) + 1
        mosaicGDEM.nlat = int((mosaicGDEM.n - mosaicGDEM.s) /
                                  mosaicGDEM.dlat) + 1

        # initialize the elev array into which data from each
        # individual tile will be added
        elev = np.zeros((mosaicGDEM.nlat, mosaicGDEM.nlon))

        # stitching all gdems into one big mosaic
        for gdem in gdems:
            lon_start = int((gdem.w - mosaicGDEM.w)/mosaicGDEM.dlon)
            lon_stop = lon_start + gdem.nlon

            lat_start = int((mosaicGDEM.n - gdem.n)/mosaicGDEM.dlat)
            lat_stop = lat_start + gdem.nlat

            elev[lat_start:lat_stop, lon_start:lon_stop] = gdem.elev

    # subsample the big mosaic to the specified extent in
    # lonmin, lonmax, latmin, latmax
    # and update the class data
    lon_start = int((lonmin - mosaicGDEM.w)/mosaicGDEM.dlon)
    lon_stop = int((lonmax - mosaicGDEM.w)/mosaicGDEM.dlon)+1

    lat_start = int((mosaicGDEM.n - latmax)/mosaicGDEM.dlat)
    lat_stop = int((mosaicGDEM.n - latmin)/mosaicGDEM.dlat)+1

    # update class data
    mosaicGDEM.e = mosaicGDEM.w + lon_stop*mosaicGDEM.dlon
    mosaicGDEM.w = mosaicGDEM.w + lon_start*mosaicGDEM.dlon
    mosaicGDEM.s = mosaicGDEM.n - lat_stop*mosaicGDEM.dlat
    mosaicGDEM.n = mosaicGDEM.n - lat_start*mosaicGDEM.dlat

    mosaicGDEM.extent = (mosaicGDEM.w, mosaicGDEM.e, mosaicGDEM.s, mosaicGDEM.n)
    mosaicGDEM.elev = elev[lat_start:lat_stop, lon_start:lon_stop]

    mosaicGDEM.nlon = mosaicGDEM.elev.shape[1]
    mosaicGDEM.nlat = mosaicGDEM.elev.shape[0]

    return mosaicGDEM

class GeoTIFF(object):
    """class for ASTER Global DEM"""

    __name__ = 'GeoTIFF'
    def __init__(self):
        self.path = None
        self.name = None
        self.w = 0.
        self.e = 0.
        self.s = 0.
        self.n = 0.
        self.extent = (0,0,0,0)
        self.dlon = 0.
        self.dlat = 0.
        self.nlon = 0
        self.nlat = 0

        self.elev = np.array([])

    def __str__(self):
        string = '\nAsterGDEM info:\n'
        if self.name:
            string += 'name: %s\n' %self.name
        string += 'west: %s\n' %self.w
        string += 'east: %s\n' %self.e
        string += 'south: %s\n' %self.s
        string += 'north: %s\n' %self.n
        string += 'lon pixel size: %s\n' %self.dlon
        string += 'lat pixel size: %s\n' %self.dlat
        string += '# of lon pixels: %s\n' %self.nlon
        string += '# of lat pixels: %s\n' %self.nlat

        return string


    def lon(self):
        return np.linspace(self.w, self.e, self.nlon)


    def lat(self):
        return np.linspace(self.n, self.s, self.nlat)


    def lonlat2d(self):
        return np.meshgrid(self.lon(),self.lat())


    def resample_slow(self, by=None, to=None, shape=None, order=3):
        """Method to resample the data either `by` a factor,
        `to` the specified spacing, or to the specified `shape`.
        by : if 1, nothing happens
             if < 1, the grid is sub-sampled by the factor
             if > 1, the grid is super-sampled byt the factor

        to : the specified spacing to which the grid is resampled to.

        shape : to the specified `shape`.

        order: int,
                The order of the spline interpolation, default is 3.
                The order has to be in the range 0-5
                0 - nearest
                1 - bi-linear
                3 - bicubic
                4 - cubic
        """

        order_dict = {  0 : 'nearest',
                        1 : 'bilinear',
                        3 : 'bicubic',
                        4 : 'cubic'    }

        if to:
            if to == self.dlon:
                return
            else:
                shape = (int(round(abs(self.nlat*self.dlat/to))),
                         int(round(abs(self.nlon*self.dlon/to))))

        elif by:
            shape = (int(round(abs(self.nlat*by))),
                     int(round(abs(self.nlon*by))))

        if shape:
            self.elev = imresize(self.elev, tuple(shape),interp=order_dict[order])
            self.nlat, self.nlon = self.elev.shape
            self.dlat, self.dlon = (self.s-self.n)/self.nlat, (self.e-self.w)/self.nlon



    def resample(self, by=None, to=None, order=3):
        """Method to resample the data either *by* a factor or
        *to* the specified spacing.
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

        if to:
            by = self.dlon/to

        if by == self.dlon:
            return

        self.elev = ndimage.zoom(self.elev,by,order=order)
        self.nlat, self.nlon = self.elev.shape
        self.dlat, self.dlon = (self.s-self.n)/self.nlat, (self.e-self.w)/self.nlon


    def elevation_profile(self, lons, lats):

        rows = []; columns = []
        for lon, lat in zip(lons, lats):
            row = (lon-self.w)/self.dlon
            column = (self.n-lat)/self.dlat

            if row > self.elev.shape[1] or column > self.elev.shape[0]:
                continue
            # Add the point to our return array
            rows += [int(row)]
            columns += [int(column)]
        return self.elev[columns, rows]


    def get_intensity(self, azimuth=315., altitude=45., scale=None, smooth=None):
        """This is a method to create an intensity array.
        It redirects to the shaded_relief fuction.
        elevation: 2-d array
        azimuth: direction of light source, degrees from north
        altitude: height of light source, degrees above the horizon
        scale: scaling value of the data
        smooth: number of cells to average before intensity calculation
        """
        return calc_intensity(self.elev, azimuth, altitude, scale, smooth)


    def write_topo_file(self, filename):
        """Write an ASCII Lon, Lat, z file for WPP topography"""

        header = '%d %d' %(self.nlon, self.nlat)
        lon,lat = self.lonlat2d()
        np.savetxt(filename, np.column_stack((lon.ravel(),
                                              lat.ravel(),
                                              self.elev.ravel())),
                   fmt='%f', header=header, comments='')


    def write_GeoTIFF(self, filename, nodata=np.nan):
        """Write a GeoTIFF file."""

        write_GeoTIFF(filename, self.elev, self.w, self.n,
                      self.dlon, self.dlat, epsg=4326,
                      nodata=nodata, rasterBand=1)
